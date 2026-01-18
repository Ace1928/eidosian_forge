import datetime
import difflib
import functools
import inspect
import json
import os
import re
import tempfile
import threading
import unittest
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch._dynamo
import torch.utils._pytree as pytree
from torch._dynamo.utils import clone_input
from torch._subclasses.schema_check_mode import SchemaCheckMode
from torch._utils_internal import get_file_path_2
from torch.overrides import TorchFunctionMode
from torch.testing._internal.optests import (
def generate_opcheck_tests(testcase: Any, namespaces: List[str], failures_dict_path: Optional[str]=None, additional_decorators: Dict[str, Callable]=None, test_utils: List[str]=DEFAULT_TEST_UTILS) -> None:
    """Given an existing TestCase, use the existing tests to generate
    additional validation tests for custom operators.

    For {all existing tests in the TestCase} x {all test utils},
    we will generate one new test. The new test runs a TorchFunctionMode
    that intercepts ``op(*args, **kwargs)`` calls and invokes
    ``test_util(op, *args, **kwargs)``, where ``op`` is an operator.

    The test_util that we support are in ALL_TEST_UTILS. They are:
    - test_schema: This runs SchemaCheckMode.
    - test_autograd_registration: This runs autograd_registration_check.
    - test_faketensor: This runs CrossRefFakeMode.
    - test_aot_dispatch_static: This runs aot_autograd_check, which:
        checks that the outputs (and gradients, if they are computable)
        are the same under eager-mode PyTorch and using AOTAutograd.
    - test_aot_dispatch_dynamic: Same as aot_dispatch_static, but
        runs AOTAutograd using dynamic shapes instead of static shapes.

    The generated test will have name ``{test_util}__{original_name}``.
    For example, if there is a method named ``test_cumsum``, then
    we will generate a ``test_schema__test_cumsum``,
    ``test_faketensor__test_cumsum``, etc.

    For more details, see https://docs.google.com/document/d/1Pj5HRZvdOq3xpFpbEjUZp2hBovhy7Wnxw14m6lF2154/edit

    Args:
        testcase: The testcase we will modify and generate additional tests for.
        namespaces: We will only intercept calls to custom operators with these
                    namespaces.
        failures_dict_path: See ``validate_failures_dict_structure`` for more details
        test_utils: a list of test_utils to generate. Example: ["test_schema", "test_faketensor"]
    """
    if additional_decorators is None:
        additional_decorators = {}
    test_methods = [m for m in dir(testcase) if m.startswith('test_') and callable(getattr(testcase, m))]
    if failures_dict_path is None:
        prev_frame = inspect.currentframe().f_back
        filename = inspect.getframeinfo(prev_frame)[0]
        failures_dict_path = get_file_path_2(os.path.dirname(filename), 'failures_dict.json')
    failures_dict = FailuresDict.load(failures_dict_path, create_file=should_update_failures_dict())
    validate_failures_dict_structure(failures_dict, test_utils, testcase)
    validate_failures_dict_formatting(failures_dict_path)

    def construct_method(attr, prefix, tester):
        method = getattr(testcase, attr)
        if getattr(method, '_torch_dont_generate_opcheck_tests', False):
            return
        new_method_name = prefix + '__' + attr

        @functools.wraps(method)
        def new_method(*args, **kwargs):
            with OpCheckMode(namespaces, prefix, tester, failures_dict, f'{testcase.__name__}.{new_method_name}', failures_dict_path):
                result = method(*args, **kwargs)
            return result
        if (pytestmark := new_method.__dict__.get('pytestmark')):
            import pytest
            opcheck_only_one = False
            for mark in pytestmark:
                if isinstance(mark, pytest.Mark) and mark.name == 'opcheck_only_one':
                    opcheck_only_one = True
            if opcheck_only_one:
                new_pytestmark = []
                for mark in pytestmark:
                    if isinstance(mark, pytest.Mark) and mark.name == 'parametrize':
                        argnames, argvalues = mark.args
                        assert not mark.kwargs, 'NYI'
                        if argnames != 'device':
                            new_pytestmark.append(pytest.mark.parametrize(argnames, (next(iter(argvalues)),)))
                            continue
                    new_pytestmark.append(mark)
                new_method.__dict__['pytestmark'] = new_pytestmark
        if new_method_name in additional_decorators:
            for dec in additional_decorators[new_method_name]:
                new_method = dec(new_method)
        if hasattr(testcase, new_method_name):
            raise RuntimeError(f'Tried to autogenerate {new_method_name} but {testcase} already has method named {new_method_name}. Please rename the original method on the TestCase.')
        setattr(testcase, new_method_name, new_method)
    test_utils = {name: ALL_TEST_UTILS[name] for name in test_utils}
    for attr in test_methods:
        for prefix, tester in test_utils.items():
            construct_method(attr, prefix, tester)
    generate_tag_tests(testcase, failures_dict, additional_decorators)
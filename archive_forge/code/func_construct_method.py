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
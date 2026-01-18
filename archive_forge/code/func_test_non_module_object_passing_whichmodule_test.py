import _collections_abc
import abc
import collections
import base64
import functools
import io
import itertools
import logging
import math
import multiprocessing
from operator import itemgetter, attrgetter
import pickletools
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
import weakref
import os
import enum
import typing
from functools import wraps
import pytest
import srsly.cloudpickle as cloudpickle
from srsly.cloudpickle.compat import pickle
from srsly.cloudpickle import register_pickle_by_value
from srsly.cloudpickle import unregister_pickle_by_value
from srsly.cloudpickle import list_registry_pickle_by_value
from srsly.cloudpickle.cloudpickle import _should_pickle_by_reference
from srsly.cloudpickle.cloudpickle import _make_empty_cell, cell_set
from srsly.cloudpickle.cloudpickle import _extract_class_dict, _whichmodule
from srsly.cloudpickle.cloudpickle import _lookup_module_and_qualname
from .testutils import subprocess_pickle_echo
from .testutils import subprocess_pickle_string
from .testutils import assert_run_python_script
from .testutils import subprocess_worker
def test_non_module_object_passing_whichmodule_test(self):

    def func(x):
        return x ** 2
    func.__module__ = None

    class NonModuleObject:

        def __ini__(self):
            self.some_attr = None

        def __getattr__(self, name):
            if name == 'func':
                return func
            else:
                raise AttributeError
    non_module_object = NonModuleObject()
    assert func(2) == 4
    assert func is non_module_object.func
    with pytest.raises(AttributeError):
        _ = non_module_object.some_attr
    try:
        sys.modules['NonModuleObject'] = non_module_object
        func_module_name = _whichmodule(func, None)
        assert func_module_name != 'NonModuleObject'
        assert func_module_name is None
        depickled_func = pickle_depickle(func, protocol=self.protocol)
        assert depickled_func(2) == 4
    finally:
        sys.modules.pop('NonModuleObject')
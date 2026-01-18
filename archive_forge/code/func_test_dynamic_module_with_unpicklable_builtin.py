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
def test_dynamic_module_with_unpicklable_builtin(self):

    class UnpickleableObject:

        def __reduce__(self):
            raise ValueError('Unpicklable object')
    mod = types.ModuleType('mod')
    exec('f = lambda x: abs(x)', mod.__dict__)
    assert mod.f(-1) == 1
    assert '__builtins__' in mod.__dict__
    unpicklable_obj = UnpickleableObject()
    with pytest.raises(ValueError):
        cloudpickle.dumps(unpicklable_obj)
    if isinstance(mod.__dict__['__builtins__'], dict):
        mod.__dict__['__builtins__']['unpickleable_obj'] = unpicklable_obj
    elif isinstance(mod.__dict__['__builtins__'], types.ModuleType):
        mod.__dict__['__builtins__'].unpickleable_obj = unpicklable_obj
    depickled_mod = pickle_depickle(mod, protocol=self.protocol)
    assert '__builtins__' in depickled_mod.__dict__
    if isinstance(depickled_mod.__dict__['__builtins__'], dict):
        assert 'abs' in depickled_mod.__builtins__
    elif isinstance(depickled_mod.__dict__['__builtins__'], types.ModuleType):
        assert hasattr(depickled_mod.__builtins__, 'abs')
    assert depickled_mod.f(-1) == 1
    assert mod.f(-1) == 1
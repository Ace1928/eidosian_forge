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
def test_closure_interacting_with_a_global_variable(self):
    global _TEST_GLOBAL_VARIABLE
    assert _TEST_GLOBAL_VARIABLE == 'default_value'
    orig_value = _TEST_GLOBAL_VARIABLE
    try:

        def f0():
            global _TEST_GLOBAL_VARIABLE
            _TEST_GLOBAL_VARIABLE = 'changed_by_f0'

        def f1():
            return _TEST_GLOBAL_VARIABLE
        cloned_f0, cloned_f1 = pickle_depickle([f0, f1], protocol=self.protocol)
        assert cloned_f0.__globals__ is cloned_f1.__globals__
        assert cloned_f0.__globals__ is not f0.__globals__
        pickled_f1 = cloudpickle.dumps(f1, protocol=self.protocol)
        cloned_f0()
        assert _TEST_GLOBAL_VARIABLE == 'default_value'
        shared_global_var = cloned_f1()
        assert shared_global_var == 'changed_by_f0', shared_global_var
        new_cloned_f1 = pickle.loads(pickled_f1)
        assert new_cloned_f1.__globals__ is not cloned_f1.__globals__
        assert new_cloned_f1.__globals__ is not f1.__globals__
        new_global_var = new_cloned_f1()
        assert new_global_var == 'default_value', new_global_var
    finally:
        _TEST_GLOBAL_VARIABLE = orig_value
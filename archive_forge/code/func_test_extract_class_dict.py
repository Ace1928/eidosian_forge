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
def test_extract_class_dict():

    class A(int):
        """A docstring"""

        def method(self):
            return 'a'

    class B:
        """B docstring"""
        B_CONSTANT = 42

        def method(self):
            return 'b'

    class C(A, B):
        C_CONSTANT = 43

        def method_c(self):
            return 'c'
    clsdict = _extract_class_dict(C)
    assert sorted(clsdict.keys()) == ['C_CONSTANT', '__doc__', 'method_c']
    assert clsdict['C_CONSTANT'] == 43
    assert clsdict['__doc__'] is None
    assert clsdict['method_c'](C()) == C().method_c()
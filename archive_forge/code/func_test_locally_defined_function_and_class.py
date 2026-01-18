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
def test_locally_defined_function_and_class(self):
    LOCAL_CONSTANT = 42

    def some_function(x, y):
        sum(range(10))
        return (x + y) / LOCAL_CONSTANT
    self.assertEqual(pickle_depickle(some_function, protocol=self.protocol)(41, 1), 1)
    self.assertEqual(pickle_depickle(some_function, protocol=self.protocol)(81, 3), 2)
    hidden_constant = lambda: LOCAL_CONSTANT

    class SomeClass:
        """Overly complicated class with nested references to symbols"""

        def __init__(self, value):
            self.value = value

        def one(self):
            return LOCAL_CONSTANT / hidden_constant()

        def some_method(self, x):
            return self.one() + some_function(x, 1) + self.value
    clone_class = pickle_depickle(SomeClass, protocol=self.protocol)
    self.assertEqual(clone_class(1).one(), 1)
    self.assertEqual(clone_class(5).some_method(41), 7)
    clone_class = subprocess_pickle_echo(SomeClass, protocol=self.protocol)
    self.assertEqual(clone_class(5).some_method(41), 7)
    self.assertEqual(pickle_depickle(SomeClass(1)).one(), 1)
    self.assertEqual(pickle_depickle(SomeClass(5)).some_method(41), 7)
    new_instance = subprocess_pickle_echo(SomeClass(5), protocol=self.protocol)
    self.assertEqual(new_instance.some_method(41), 7)
    self.assertEqual(pickle_depickle(SomeClass(1).one)(), 1)
    self.assertEqual(pickle_depickle(SomeClass(5).some_method)(41), 7)
    new_method = subprocess_pickle_echo(SomeClass(5).some_method, protocol=self.protocol)
    self.assertEqual(new_method(41), 7)
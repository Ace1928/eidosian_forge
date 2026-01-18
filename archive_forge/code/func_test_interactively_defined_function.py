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
@pytest.mark.skipif(platform.python_implementation() == 'PyPy', reason='fails sometimes for pypy on conda-forge')
def test_interactively_defined_function(self):
    code = "        from srsly.tests.cloudpickle.testutils import subprocess_pickle_echo\n\n        CONSTANT = 42\n\n        class Foo(object):\n\n            def method(self, x):\n                return x\n\n        foo = Foo()\n\n        def f0(x):\n            return x ** 2\n\n        def f1():\n            return Foo\n\n        def f2(x):\n            return Foo().method(x)\n\n        def f3():\n            return Foo().method(CONSTANT)\n\n        def f4(x):\n            return foo.method(x)\n\n        def f5(x):\n            # Recursive call to a dynamically defined function.\n            if x <= 0:\n                return f4(x)\n            return f5(x - 1) + 1\n\n        cloned = subprocess_pickle_echo(lambda x: x**2, protocol={protocol})\n        assert cloned(3) == 9\n\n        cloned = subprocess_pickle_echo(f0, protocol={protocol})\n        assert cloned(3) == 9\n\n        cloned = subprocess_pickle_echo(Foo, protocol={protocol})\n        assert cloned().method(2) == Foo().method(2)\n\n        cloned = subprocess_pickle_echo(Foo(), protocol={protocol})\n        assert cloned.method(2) == Foo().method(2)\n\n        cloned = subprocess_pickle_echo(f1, protocol={protocol})\n        assert cloned()().method('a') == f1()().method('a')\n\n        cloned = subprocess_pickle_echo(f2, protocol={protocol})\n        assert cloned(2) == f2(2)\n\n        cloned = subprocess_pickle_echo(f3, protocol={protocol})\n        assert cloned() == f3()\n\n        cloned = subprocess_pickle_echo(f4, protocol={protocol})\n        assert cloned(2) == f4(2)\n\n        cloned = subprocess_pickle_echo(f5, protocol={protocol})\n        assert cloned(7) == f5(7) == 7\n        ".format(protocol=self.protocol)
    assert_run_python_script(textwrap.dedent(code))
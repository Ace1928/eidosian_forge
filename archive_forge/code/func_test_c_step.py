from __future__ import absolute_import
import os
import re
import sys
import trace
import inspect
import warnings
import unittest
import textwrap
import tempfile
import functools
import traceback
import itertools
import gdb
from .. import libcython
from .. import libpython
from . import TestLibCython as test_libcython
from ...Utils import add_metaclass
def test_c_step(self):
    self.break_and_run('some_c_function()')
    gdb.execute('cy step', to_string=True)
    self.assertEqual(gdb.selected_frame().name(), 'some_c_function')
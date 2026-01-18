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
def test_CythonFunction(self):
    self.assertEqual(self.spam_func.qualified_name, 'codefile.spam')
    self.assertEqual(self.spam_meth.qualified_name, 'codefile.SomeClass.spam')
    self.assertEqual(self.spam_func.module, self.module)
    assert self.eggs_func.pf_cname, (self.eggs_func, self.eggs_func.pf_cname)
    assert not self.ham_func.pf_cname
    assert not self.spam_func.pf_cname
    assert not self.spam_meth.pf_cname
    self.assertEqual(self.spam_func.type, libcython.CObject)
    self.assertEqual(self.ham_func.type, libcython.CObject)
    self.assertEqual(self.spam_func.arguments, ['a'])
    self.assertEqual(self.spam_func.step_into_functions, {'puts', 'some_c_function'})
    expected_lineno = test_libcython.source_to_lineno['def spam(a=0):']
    self.assertEqual(self.spam_func.lineno, expected_lineno)
    self.assertEqual(sorted(self.spam_func.locals), list('abcd'))
import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def assertMemRChr(self, expected, s, c):
    from .._dirstate_helpers_pyx import _py_memrchr
    self.assertEqual(expected, _py_memrchr(s, c))
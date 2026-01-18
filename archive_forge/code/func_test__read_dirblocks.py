import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test__read_dirblocks(self):
    if compiled_dirstate_helpers_feature.available():
        from .._dirstate_helpers_pyx import _read_dirblocks
    else:
        from .._dirstate_helpers_py import _read_dirblocks
    self.assertIs(_read_dirblocks, dirstate._read_dirblocks)
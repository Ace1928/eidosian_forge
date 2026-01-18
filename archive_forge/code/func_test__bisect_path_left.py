import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test__bisect_path_left(self):
    if compiled_dirstate_helpers_feature.available():
        from .._dirstate_helpers_pyx import _bisect_path_left
    else:
        from .._dirstate_helpers_py import _bisect_path_left
    self.assertIs(_bisect_path_left, dirstate._bisect_path_left)
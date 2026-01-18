import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def do_update_entry(self, state, entry, abspath):
    stat_value = os.lstat(abspath)
    return self.update_entry(state, entry, abspath, stat_value)
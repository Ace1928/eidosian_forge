import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def get_state_with_a(self):
    """Create a DirState tracking a single object named 'a'"""
    state = test_dirstate.InstrumentedDirState.initialize('dirstate')
    self.addCleanup(state.unlock)
    state.add('a', b'a-id', 'file', None, b'')
    entry = state._get_entry(0, path_utf8=b'a')
    return (state, entry)
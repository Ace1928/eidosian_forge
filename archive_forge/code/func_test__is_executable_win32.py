import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test__is_executable_win32(self):
    state, entry = self.get_state_with_a()
    self.build_tree(['a'])
    state._use_filesystem_for_exec = False
    entry[1][0] = (b'f', b'', 0, True, dirstate.DirState.NULLSTAT)
    stat_value = os.lstat('a')
    packed_stat = dirstate.pack_stat(stat_value)
    state.adjust_time(-10)
    self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
    self.assertEqual([(b'f', b'', 14, True, dirstate.DirState.NULLSTAT)], entry[1])
    state.adjust_time(+20)
    digest = b'b50e5406bb5e153ebbeb20268fcf37c87e1ecfb6'
    self.update_entry(state, entry, abspath=b'a', stat_value=stat_value)
    self.assertEqual([(b'f', b'', 14, True, dirstate.DirState.NULLSTAT)], entry[1])
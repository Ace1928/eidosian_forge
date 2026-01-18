import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_discard_renamed(self):
    null_stat = dirstate.DirState.NULLSTAT
    present_dir = (b'd', b'', 0, False, null_stat)
    present_file = (b'f', b'', 0, False, null_stat)
    absent = dirstate.DirState.NULL_PARENT_DETAILS
    root_key = (b'', b'', b'a-root-value')
    file_in_root_key = (b'', b'file-in-root', b'a-file-id')
    file_rename_s_key = (b'', b'file-s', b'b-file-id')
    file_rename_t_key = (b'', b'file-t', b'b-file-id')
    key_in_1 = (b'', b'file-in-1', b'c-file-id')
    key_in_2 = (b'', b'file-in-2', b'c-file-id')
    dirblocks = [(b'', [(root_key, [present_dir, present_dir, present_dir])]), (b'', [(key_in_1, [absent, present_file, (b'r', b'file-in-2', b'c-file-id')]), (key_in_2, [absent, (b'r', b'file-in-1', b'c-file-id'), present_file]), (file_in_root_key, [present_file, present_file, present_file]), (file_rename_s_key, [(b'r', b'file-t', b'b-file-id'), absent, present_file]), (file_rename_t_key, [present_file, absent, (b'r', b'file-s', b'b-file-id')])])]
    exp_dirblocks = [(b'', [(root_key, [present_dir, present_dir])]), (b'', [(key_in_1, [absent, present_file]), (file_in_root_key, [present_file, present_file]), (file_rename_t_key, [present_file, absent])])]
    state = self.create_empty_dirstate()
    self.addCleanup(state.unlock)
    state._set_data([b'parent-id', b'merged-id'], dirblocks[:])
    state._validate()
    state._discard_merge_parents()
    state._validate()
    self.assertEqual(exp_dirblocks, state._dirblocks)
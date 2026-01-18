import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def create_dirstate_with_root_and_subdir(self):
    """Return a locked DirState with a root and a subdir"""
    packed_stat = b'AAAAREUHaIpFB2iKAAADAQAtkqUAAIGk'
    subdir_entry = ((b'', b'subdir', b'subdir-id'), [(b'd', b'', 0, False, packed_stat)])
    state = self.create_dirstate_with_root()
    try:
        dirblocks = list(state._dirblocks)
        dirblocks[1][1].append(subdir_entry)
        state._set_data([], dirblocks)
    except:
        state.unlock()
        raise
    return state
import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def create_empty_dirstate(self):
    fd, path = tempfile.mkstemp(prefix='bzr-dirstate')
    self.addCleanup(os.remove, path)
    os.close(fd)
    state = dirstate.DirState.initialize(path)
    self.addCleanup(state.unlock)
    return state
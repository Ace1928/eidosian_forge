import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def create_updated_dirstate(self):
    self.build_tree(['a-file'])
    tree = self.make_branch_and_tree('.')
    tree.add(['a-file'], ids=[b'a-id'])
    tree.commit('add a-file')
    state = dirstate.DirState.from_tree(tree, 'dirstate')
    state.save()
    state.unlock()
    state = dirstate.DirState.on_file('dirstate')
    state.lock_read()
    return state
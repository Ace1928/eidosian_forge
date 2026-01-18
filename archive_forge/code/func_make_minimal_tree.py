import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def make_minimal_tree(self):
    tree1 = self.make_branch_and_memory_tree('tree1')
    tree1.lock_write()
    self.addCleanup(tree1.unlock)
    tree1.add('')
    revid1 = tree1.commit('foo')
    return (tree1, revid1)
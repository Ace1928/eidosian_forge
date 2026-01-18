import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_colliding_fileids(self):
    parents = []
    for i in range(7):
        tree = self.make_branch_and_tree('tree%d' % i)
        self.build_tree(['tree%d/name' % i])
        tree.add(['name'], ids=[b'file-id%d' % i])
        revision_id = b'revid-%d' % i
        tree.commit('message', rev_id=revision_id)
        parents.append((revision_id, tree.branch.repository.revision_tree(revision_id)))
    state = dirstate.DirState.initialize('dirstate')
    try:
        state.set_parent_trees(parents, [])
        state._validate()
    finally:
        state.unlock()
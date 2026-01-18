import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_bisect_dirblocks(self):
    tree, state, expected = self.create_duplicated_dirstate()
    self.assertBisectDirBlocks(expected, [[b'', b'a', b'a2', b'b', b'b2', b'b-c', b'b-c2', b'f', b'f2']], state, [b''])
    self.assertBisectDirBlocks(expected, [[b'b/c', b'b/c2', b'b/d', b'b/d2']], state, [b'b'])
    self.assertBisectDirBlocks(expected, [[b'b/d/e', b'b/d/e2']], state, [b'b/d'])
    self.assertBisectDirBlocks(expected, [[b'', b'a', b'a2', b'b', b'b2', b'b-c', b'b-c2', b'f', b'f2'], [b'b/c', b'b/c2', b'b/d', b'b/d2'], [b'b/d/e', b'b/d/e2']], state, [b'', b'b', b'b/d'])
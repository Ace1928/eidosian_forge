import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_bisect_dirblocks_missing(self):
    tree, state, expected = self.create_basic_dirstate()
    self.assertBisectDirBlocks(expected, [[b'b/d/e'], None], state, [b'b/d', b'b/e'])
    self.assertBisectDirBlocks(expected, [None], state, [b'a'])
    self.assertBisectDirBlocks(expected, [None], state, [b'b/c'])
    self.assertBisectDirBlocks(expected, [None], state, [b'c'])
    self.assertBisectDirBlocks(expected, [None], state, [b'b/d/e'])
    self.assertBisectDirBlocks(expected, [None], state, [b'f'])
import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_bisect_recursive_renamed(self):
    tree, state, expected = self.create_renamed_dirstate()
    self.assertBisectRecursive(expected, [b'a', b'b/g'], state, [b'a'])
    self.assertBisectRecursive(expected, [b'a', b'b/g'], state, [b'b/g'])
    self.assertBisectRecursive(expected, [b'a', b'b', b'b/c', b'b/d', b'b/d/e', b'b/g', b'h', b'h/e'], state, [b'b'])
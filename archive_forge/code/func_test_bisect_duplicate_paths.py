import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_bisect_duplicate_paths(self):
    """When bisecting for a path, handle multiple entries."""
    tree, state, expected = self.create_duplicated_dirstate()
    self.assertBisect(expected, [[b'']], state, [b''])
    self.assertBisect(expected, [[b'a', b'a2']], state, [b'a'])
    self.assertBisect(expected, [[b'b', b'b2']], state, [b'b'])
    self.assertBisect(expected, [[b'b/c', b'b/c2']], state, [b'b/c'])
    self.assertBisect(expected, [[b'b/d', b'b/d2']], state, [b'b/d'])
    self.assertBisect(expected, [[b'b/d/e', b'b/d/e2']], state, [b'b/d/e'])
    self.assertBisect(expected, [[b'b-c', b'b-c2']], state, [b'b-c'])
    self.assertBisect(expected, [[b'f', b'f2']], state, [b'f'])
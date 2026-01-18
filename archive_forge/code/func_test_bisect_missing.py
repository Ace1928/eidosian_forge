import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_bisect_missing(self):
    """Test that bisect return None if it cannot find a path."""
    tree, state, expected = self.create_basic_dirstate()
    self.assertBisect(expected, [None], state, [b'foo'])
    self.assertBisect(expected, [None], state, [b'b/foo'])
    self.assertBisect(expected, [None], state, [b'bar/foo'])
    self.assertBisect(expected, [None], state, [b'b-c/foo'])
    self.assertBisect(expected, [[b'a'], None, [b'b/d']], state, [b'a', b'foo', b'b/d'])
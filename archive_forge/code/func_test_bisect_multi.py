import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_bisect_multi(self):
    """Bisect can be used to find multiple records at the same time."""
    tree, state, expected = self.create_basic_dirstate()
    self.assertBisect(expected, [[b'a'], [b'b'], [b'f']], state, [b'a', b'b', b'f'])
    self.assertBisect(expected, [[b'f'], [b'b/d'], [b'b/d/e']], state, [b'f', b'b/d', b'b/d/e'])
    self.assertBisect(expected, [[b'b'], [b'b-c'], [b'b/c']], state, [b'b', b'b-c', b'b/c'])
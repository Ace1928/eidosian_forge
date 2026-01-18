import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_defaultsha1provider_sha1(self):
    text = b'test\r\nwith\nall\rpossible line endings\r\n'
    self.build_tree_contents([('foo', text)])
    expected_sha = osutils.sha_string(text)
    p = dirstate.DefaultSHA1Provider()
    self.assertEqual(expected_sha, p.sha1('foo'))
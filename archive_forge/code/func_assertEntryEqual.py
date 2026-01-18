import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def assertEntryEqual(self, dirname, basename, file_id, state, path, index):
    """Check that the right entry is returned for a request to getEntry."""
    entry = state._get_entry(index, path_utf8=path)
    if file_id is None:
        self.assertEqual((None, None), entry)
    else:
        cur = entry[0]
        self.assertEqual((dirname, basename, file_id), cur[:3])
import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_dirblock_missing_rename(self):
    tree, state, expected = self.create_renamed_dirstate()
    state._read_dirblocks_if_needed()
    last_dirblock = state._dirblocks[-1]
    last_dirblock[1].append(((b'h', b'z', b'a-id'), [(b'a', b'', 0, False, b''), (b'a', b'', 0, False, b'')]))
    e = self.assertRaises(AssertionError, state._validate)
    self.assertContainsRe(str(e), 'file a-id is absent in row')
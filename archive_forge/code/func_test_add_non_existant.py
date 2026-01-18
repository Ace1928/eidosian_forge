import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_add_non_existant(self):
    """Test smart-adding a file that does not exist."""
    wt = self.make_branch_and_tree('.')
    self.assertRaises(transport.NoSuchFile, wt.smart_add, ['non-existant-file'])
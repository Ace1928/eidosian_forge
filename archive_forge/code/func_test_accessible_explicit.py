import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_accessible_explicit(self):
    osutils.normalized_filename = osutils._accessible_normalized_filename
    if self.workingtree_format.requires_normalized_unicode_filenames:
        raise tests.TestNotApplicable('Working tree format smart_add requires normalized unicode filenames')
    self.wt.smart_add(['å'])
    self.wt.lock_read()
    self.addCleanup(self.wt.unlock)
    self.assertEqual([('', 'directory'), ('å', 'file')], [(path, ie.kind) for path, ie in self.wt.iter_entries_by_dir()])
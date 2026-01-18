import errno
import os
import sys
import time
from io import BytesIO
from breezy.bzr.transform import resolve_checkout
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from ... import osutils, tests, trace, transform, urlutils
from ...bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ...errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ...osutils import file_kind, pathjoin
from ...transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ...transport import FileExists
from ...tree import TreeChange
from .. import TestSkipped, features
from ..features import HardlinkFeature, SymlinkFeature
def test_iter_changes_new(self):
    if self.wt.supports_setting_file_ids():
        root_id = self.wt.path2id('')
    transform, root = self.transform()
    transform.new_file('old', root, [b'blah'])
    transform.apply()
    transform, root = self.transform()
    try:
        old = transform.trans_id_tree_path('old')
        transform.version_file(old, file_id=b'id-1')
        changes = list(transform.iter_changes())
        self.assertEqual(1, len(changes))
        self.assertEqual((None, 'old'), changes[0].path)
        self.assertEqual(False, changes[0].changed_content)
        self.assertEqual((False, True), changes[0].versioned)
        self.assertEqual((False, False), changes[0].executable)
        if self.wt.supports_setting_file_ids():
            self.assertEqual((root_id, root_id), changes[0].parent_id)
            self.assertEqual(b'id-1', changes[0].file_id)
    finally:
        transform.finalize()
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
def test_iter_changes_move_missing(self):
    """Test moving ids with no files around"""
    transform, root = self.transform()
    transform.new_directory('floater', root, b'floater-id')
    transform.apply()
    transform, root = self.transform()
    transform.delete_contents(transform.trans_id_tree_path('floater'))
    transform.apply()
    transform, root = self.transform()
    floater = transform.trans_id_tree_path('floater')
    try:
        transform.adjust_path('flitter', root, floater)
        if self.wt.has_versioned_directories():
            self.assertTreeChanges(transform, [TreeChange(('floater', 'flitter'), False, (True, True), ('floater', 'flitter'), (None, None), (False, False), False)])
        else:
            self.assertTreeChanges(transform, [])
    finally:
        transform.finalize()
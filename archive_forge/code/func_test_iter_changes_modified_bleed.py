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
def test_iter_changes_modified_bleed(self):
    """Modified flag should not bleed from one change to another"""
    transform, root = self.transform()
    transform.new_file('file1', root, [b'blah'], b'id-1')
    transform.new_file('file2', root, [b'blah'], b'id-2')
    transform.apply()
    transform, root = self.transform()
    try:
        transform.delete_contents(transform.trans_id_tree_path('file1'))
        transform.set_executability(True, transform.trans_id_tree_path('file2'))
        self.assertTreeChanges(transform, [TreeChange(('file1', 'file1'), True, (True, True), ('file1', 'file1'), ('file', None), (False, False), False), TreeChange(('file2', 'file2'), False, (True, True), ('file2', 'file2'), ('file', 'file'), (False, True), False)])
    finally:
        transform.finalize()
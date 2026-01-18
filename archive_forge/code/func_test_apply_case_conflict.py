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
def test_apply_case_conflict(self):
    """Ensure that a transform with case conflicts can always be applied"""
    tree = self.make_branch_and_tree('tree')
    transform = tree.transform()
    self.addCleanup(transform.finalize)
    transform.new_file('file', transform.root, [b'content'])
    transform.new_file('FiLe', transform.root, [b'content'])
    dir = transform.new_directory('dir', transform.root)
    transform.new_file('dirfile', dir, [b'content'])
    transform.new_file('dirFiLe', dir, [b'content'])
    resolve_conflicts(transform)
    transform.apply()
    self.assertPathExists('tree/file')
    if not os.path.exists('tree/FiLe.moved'):
        self.assertPathExists('tree/FiLe')
    self.assertPathExists('tree/dir/dirfile')
    if not os.path.exists('tree/dir/dirFiLe.moved'):
        self.assertPathExists('tree/dir/dirFiLe')
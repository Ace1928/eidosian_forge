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
def test_case_insensitive_limbo(self):
    tree = self.make_branch_and_tree('tree')
    tree.case_sensitive = False
    transform = tree.transform()
    self.addCleanup(transform.finalize)
    dir = transform.new_directory('dir', transform.root)
    first = transform.new_file('file', dir, [b'content'])
    second = transform.new_file('FiLe', dir, [b'content'])
    self.assertContainsRe(transform._limbo_name(first), 'new-1/file')
    self.assertNotContainsRe(transform._limbo_name(second), 'new-1/FiLe')
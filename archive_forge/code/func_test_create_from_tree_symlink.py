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
def test_create_from_tree_symlink(self):
    self.requireFeature(SymlinkFeature(self.test_dir))
    tree1 = self.make_branch_and_tree('tree1')
    os.symlink('bar', 'tree1/foo')
    tree1.add('foo')
    tt = self.make_branch_and_tree('tree2').transform()
    foo_trans_id = tt.create_path('foo', tt.root)
    create_from_tree(tt, foo_trans_id, tree1, 'foo')
    tt.apply()
    self.assertEqual('bar', os.readlink('tree2/foo'))
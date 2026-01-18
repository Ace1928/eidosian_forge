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
def test_apply_informs_tree_of_observed_sha1(self):
    trans, root, contents, sha1 = self.transform_for_sha1_test()
    from ...bzr.workingtree import InventoryWorkingTree
    if not isinstance(self.wt, InventoryWorkingTree):
        self.skipTest('not a bzr working tree')
    trans_id = trans.new_file('file1', root, contents, file_id=b'file1-id', sha1=sha1)
    calls = []
    orig = self.wt._observed_sha1

    def _observed_sha1(*args):
        calls.append(args)
        orig(*args)
    self.wt._observed_sha1 = _observed_sha1
    trans.apply()
    self.assertEqual([('file1', trans._observed_sha1s[trans_id])], calls)
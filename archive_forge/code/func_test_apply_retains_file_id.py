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
def test_apply_retains_file_id(self):
    transform, root = self.transform()
    if not self.wt.supports_setting_file_ids():
        self.skipTest('format does not support file ids')
    old_root_id = transform.tree_file_id(root)
    transform.unversion_file(root)
    transform.apply()
    self.assertEqual(old_root_id, self.wt.path2id(''))
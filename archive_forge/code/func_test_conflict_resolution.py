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
def test_conflict_resolution(self):
    conflicts, emerald, oz, old_dorothy, new_dorothy, munchkincity = self.get_conflicted()
    resolve_conflicts(conflicts)
    self.assertEqual(conflicts.final_name(old_dorothy), 'dorothy.moved')
    if self.wt.supports_setting_file_ids():
        self.assertIs(conflicts.final_file_id(old_dorothy), None)
        self.assertEqual(conflicts.final_file_id(new_dorothy), b'dorothy-id')
    self.assertEqual(conflicts.final_name(new_dorothy), 'dorothy')
    self.assertEqual(conflicts.final_parent(emerald), oz)
    conflicts.apply()
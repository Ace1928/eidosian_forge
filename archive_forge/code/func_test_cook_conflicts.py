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
def test_cook_conflicts(self):
    tt, emerald, oz, old_dorothy, new_dorothy, munchkincity = self.get_conflicted()
    raw_conflicts = resolve_conflicts(tt)
    cooked_conflicts = list(tt.cook_conflicts(raw_conflicts))
    if self.wt.supports_setting_file_ids():
        duplicate = DuplicateEntry('Moved existing file to', 'dorothy.moved', 'dorothy', None, b'dorothy-id')
        self.assertEqual(cooked_conflicts[0], duplicate)
        duplicate_id = DuplicateID('Unversioned existing file', 'dorothy.moved', 'dorothy', None, b'dorothy-id')
        self.assertEqual(cooked_conflicts[1], duplicate_id)
        missing_parent = MissingParent('Created directory', 'munchkincity', b'munchkincity-id')
        deleted_parent = DeletingParent('Not deleting', 'oz', b'oz-id')
        self.assertEqual(cooked_conflicts[2], missing_parent)
        unversioned_parent = UnversionedParent('Versioned directory', 'munchkincity', b'munchkincity-id')
        unversioned_parent2 = UnversionedParent('Versioned directory', 'oz', b'oz-id')
        self.assertEqual(cooked_conflicts[3], unversioned_parent)
        parent_loop = ParentLoop('Cancelled move', 'oz/emeraldcity', 'oz/emeraldcity', b'emerald-id', b'emerald-id')
        self.assertEqual(cooked_conflicts[4], deleted_parent)
        self.assertEqual(cooked_conflicts[5], unversioned_parent2)
        self.assertEqual(cooked_conflicts[6], parent_loop)
        self.assertEqual(len(cooked_conflicts), 7)
    elif self.wt.has_versioned_directories():
        self.assertEqual({c.path for c in cooked_conflicts}, {'oz/emeraldcity', 'oz', 'munchkincity', 'dorothy.moved'})
    else:
        self.assertEqual({c.path for c in cooked_conflicts}, {'oz/emeraldcity', 'oz', 'dorothy.moved'})
    tt.finalize()
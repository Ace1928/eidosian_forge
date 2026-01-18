import os
from io import BytesIO
from .. import (conflicts, errors, symbol_versioning, trace, transport,
from ..bzr import bzrdir
from ..bzr import conflicts as _mod_bzr_conflicts
from ..bzr import workingtree as bzrworkingtree
from ..bzr import workingtree_3, workingtree_4
from ..lock import write_locked
from ..lockdir import LockDir
from ..tree import TreeDirectory, TreeEntry, TreeFile, TreeLink
from . import TestCase, TestCaseWithTransport, TestSkipped
from .features import SymlinkFeature
def test_auto_resolve_missing(self):
    tree = self.make_branch_and_tree('tree')
    file_conflict = _mod_bzr_conflicts.TextConflict('hello', b'hello-id')
    tree.set_conflicts([file_conflict])
    remaining, resolved = tree.auto_resolve()
    self.assertEqual(remaining, [])
    self.assertEqual(resolved, conflicts.ConflictList([_mod_bzr_conflicts.TextConflict('hello', 'hello-id')]))
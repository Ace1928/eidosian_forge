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
def test_direct_subtree(self):
    tree = self.make_simple_tree()
    self.make_branch_and_tree('tree/a/b')
    self.assertEqual([('directory', b'root-id'), ('directory', b'a-id'), ('tree-reference', b'b-id')], [(ie.kind, ie.file_id) for path, ie in tree.iter_entries_by_dir()])
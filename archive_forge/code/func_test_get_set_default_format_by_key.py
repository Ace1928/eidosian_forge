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
def test_get_set_default_format_by_key(self):
    old_format = workingtree.format_registry.get_default()
    format = SampleTreeFormat()
    workingtree.format_registry.register(format)
    self.addCleanup(workingtree.format_registry.remove, format)
    self.assertTrue(isinstance(old_format, workingtree_4.WorkingTreeFormat6))
    workingtree.format_registry.set_default_key(format.get_format_string())
    try:
        dir = bzrdir.BzrDirMetaFormat1().initialize('.')
        dir.create_repository()
        dir.create_branch()
        result = dir.create_workingtree()
        self.assertEqual(result, 'A tree')
    finally:
        workingtree.format_registry.set_default_key(old_format.get_format_string())
    self.assertEqual(old_format, workingtree.format_registry.get_default())
import os
from io import BytesIO
import breezy
from .. import config, controldir, errors, trace
from .. import transport as _mod_transport
from ..branch import Branch
from ..bzr.bzrdir import BzrDirMetaFormat1
from ..commit import (CannotCommitSelectedFileMerge, Commit,
from ..errors import BzrError, LockContention
from ..tree import TreeChange
from . import TestCase, TestCaseWithTransport, test_foreign
from .features import SymlinkFeature
from .matchers import MatchesAncestry, MatchesTreeChanges
def test_commit_removals_respects_filespec(self):
    """Commit respects the specified_files for removals."""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b'])
    tree.add(['a', 'b'])
    tree.commit('added a, b')
    tree.remove(['a', 'b'])
    tree.commit('removed a', specific_files='a')
    basis = tree.basis_tree()
    with tree.lock_read():
        self.assertFalse(basis.is_versioned('a'))
        self.assertTrue(basis.is_versioned('b'))
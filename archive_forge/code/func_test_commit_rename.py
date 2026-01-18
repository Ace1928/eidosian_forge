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
def test_commit_rename(self):
    """Test commit of a revision where a file is renamed."""
    tree = self.make_branch_and_tree('.')
    b = tree.branch
    self.build_tree(['hello'], line_endings='binary')
    tree.add(['hello'], ids=[b'hello-id'])
    tree.commit(message='one', rev_id=b'test@rev-1', allow_pointless=False)
    tree.rename_one('hello', 'fruity')
    tree.commit(message='renamed', rev_id=b'test@rev-2', allow_pointless=False)
    eq = self.assertEqual
    tree1 = b.repository.revision_tree(b'test@rev-1')
    tree1.lock_read()
    self.addCleanup(tree1.unlock)
    eq(tree1.id2path(b'hello-id'), 'hello')
    eq(tree1.get_file_text('hello'), b'contents of hello\n')
    self.assertFalse(tree1.has_filename('fruity'))
    self.check_tree_shape(tree1, ['hello'])
    eq(tree1.get_file_revision('hello'), b'test@rev-1')
    tree2 = b.repository.revision_tree(b'test@rev-2')
    tree2.lock_read()
    self.addCleanup(tree2.unlock)
    eq(tree2.id2path(b'hello-id'), 'fruity')
    eq(tree2.get_file_text('fruity'), b'contents of hello\n')
    self.check_tree_shape(tree2, ['fruity'])
    eq(tree2.get_file_revision('fruity'), b'test@rev-2')
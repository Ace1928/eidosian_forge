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
def test_commit_callback(self):
    """Commit should invoke a callback to get the message"""
    tree = self.make_branch_and_tree('.')
    try:
        tree.commit()
    except Exception as e:
        self.assertTrue(isinstance(e, BzrError))
        self.assertEqual('The message or message_callback keyword parameter is required for commit().', str(e))
    else:
        self.fail('exception not raised')
    cb = self.Callback('commit 1', self)
    tree.commit(message_callback=cb)
    self.assertTrue(cb.called)
    repository = tree.branch.repository
    message = repository.get_revision(tree.last_revision()).message
    self.assertEqual('commit 1', message)
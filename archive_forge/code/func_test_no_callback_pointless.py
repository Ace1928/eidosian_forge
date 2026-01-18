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
def test_no_callback_pointless(self):
    """Callback should not be invoked for pointless commit"""
    tree = self.make_branch_and_tree('.')
    cb = self.Callback('commit 2', self)
    self.assertRaises(PointlessCommit, tree.commit, message_callback=cb, allow_pointless=False)
    self.assertFalse(cb.called)
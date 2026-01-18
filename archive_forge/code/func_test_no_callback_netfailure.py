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
def test_no_callback_netfailure(self):
    """Callback should not be invoked if connectivity fails"""
    tree = self.make_branch_and_tree('.')
    cb = self.Callback('commit 2', self)
    repository = tree.branch.repository

    def raise_(self, arg, arg2, arg3=None, arg4=None):
        raise _mod_transport.NoSuchFile('foo')
    repository.add_inventory = raise_
    repository.add_inventory_by_delta = raise_
    self.assertRaises(_mod_transport.NoSuchFile, tree.commit, message_callback=cb)
    self.assertFalse(cb.called)
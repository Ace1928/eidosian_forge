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
def test_commit_invokes_hooks(self):
    import breezy.commit as commit
    wt = self.make_branch_and_tree('.')
    branch = wt.branch
    calls = []

    def called(branch, rev_id):
        calls.append('called')
    breezy.ahook = called
    try:
        conf = config.MemoryStack(b'post_commit=breezy.ahook breezy.ahook')
        commit.Commit(config_stack=conf).commit(message='base', allow_pointless=True, rev_id=b'A', working_tree=wt)
        self.assertEqual(['called', 'called'], calls)
    finally:
        del breezy.ahook
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
def test_commit_failed_signature(self):
    import breezy.commit as commit
    import breezy.gpg
    oldstrategy = breezy.gpg.GPGStrategy
    wt = self.make_branch_and_tree('.')
    branch = wt.branch
    wt.commit('base', allow_pointless=True, rev_id=b'A')
    self.assertFalse(branch.repository.has_signature_for_revision_id(b'A'))
    try:
        breezy.gpg.GPGStrategy = breezy.gpg.DisabledGPGStrategy
        conf = config.MemoryStack(b'\ncreate_signatures=always\n')
        self.assertRaises(breezy.gpg.SigningFailed, commit.Commit(config_stack=conf).commit, message='base', allow_pointless=True, rev_id=b'B', working_tree=wt)
        branch = Branch.open(self.get_url('.'))
        self.assertEqual(branch.last_revision(), b'A')
        self.assertFalse(branch.repository.has_revision(b'B'))
    finally:
        breezy.gpg.GPGStrategy = oldstrategy
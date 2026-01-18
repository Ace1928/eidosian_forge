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
def test_commit_saves_1ms_timestamp(self):
    """Passing in a timestamp is saved with 1ms resolution"""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a'])
    tree.add('a')
    tree.commit('added a', timestamp=1153248633.418672, timezone=0, rev_id=b'a1')
    rev = tree.branch.repository.get_revision(b'a1')
    self.assertEqual(1153248633.419, rev.timestamp)
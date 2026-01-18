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
def test_commit_bound_lossy_foreign(self):
    """Attempt a lossy commit to a bzr branch bound to a foreign branch."""
    test_foreign.register_dummy_foreign_for_test(self)
    foreign_branch = self.make_branch('foreign', format=test_foreign.DummyForeignVcsDirFormat())
    wt = foreign_branch.create_checkout('local')
    b = wt.branch
    with open('local/hello', 'w') as f:
        f.write('hello world')
    wt.add('hello')
    revid = wt.commit(message='add hello', lossy=True, timestamp=1302659388, timezone=0)
    self.assertEqual(b'dummy-v1:1302659388-0-0', revid)
    self.assertEqual(b'dummy-v1:1302659388-0-0', foreign_branch.last_revision())
    self.assertEqual(b'dummy-v1:1302659388-0-0', wt.branch.last_revision())
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
def test_missing_commit(self):
    """Test a commit with a missing file"""
    wt = self.make_branch_and_tree('.')
    b = wt.branch
    with open('hello', 'w') as f:
        f.write('hello world')
    wt.add(['hello'], ids=[b'hello-id'])
    wt.commit(message='add hello')
    os.remove('hello')
    reporter = CapturingReporter()
    wt.commit('removed hello', rev_id=b'rev2', reporter=reporter)
    self.assertEqual([('missing', 'hello'), ('deleted', 'hello')], reporter.calls)
    tree = b.repository.revision_tree(b'rev2')
    self.assertFalse(tree.has_filename('hello'))
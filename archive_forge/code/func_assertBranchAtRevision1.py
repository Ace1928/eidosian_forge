from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
def assertBranchAtRevision1(params):
    self.assertEqual((0, revision.NULL_REVISION), params.branch.last_revision_info())
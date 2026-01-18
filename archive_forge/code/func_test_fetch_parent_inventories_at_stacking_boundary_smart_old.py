import sys
from breezy import errors, osutils, repository
from breezy.bzr import inventory, versionedfile
from breezy.bzr.vf_search import SearchResult
from breezy.errors import NoSuchRevision
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable
from breezy.tests.per_interrepository import TestCaseWithInterRepository
from breezy.tests.per_interrepository.test_interrepository import \
def test_fetch_parent_inventories_at_stacking_boundary_smart_old(self):
    self.setup_smart_server_with_call_log()
    self.disable_verb(b'Repository.insert_stream_1.19')
    try:
        self.test_fetch_parent_inventories_at_stacking_boundary()
    except errors.ConnectionReset:
        self.knownFailure('Random spurious failure, see bug 874153')
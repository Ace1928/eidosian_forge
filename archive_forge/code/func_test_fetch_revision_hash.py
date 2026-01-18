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
def test_fetch_revision_hash(self):
    """Ensure that inventory hashes are updated by fetch"""
    if not self.repository_format_to.supports_full_versioned_files:
        raise TestNotApplicable('Need full versioned files')
    from_tree = self.make_branch_and_tree('tree')
    revid = from_tree.commit('foo')
    to_repo = self.make_to_repository('to')
    try:
        to_repo.fetch(from_tree.branch.repository)
    except errors.NoRoundtrippingSupport:
        raise TestNotApplicable('roundtripping not supported')
    recorded_inv_sha1 = to_repo.get_revision(revid).inventory_sha1
    to_repo.lock_read()
    self.addCleanup(to_repo.unlock)
    stream = to_repo.inventories.get_record_stream([(revid,)], 'unordered', True)
    bytes = next(stream).get_bytes_as('fulltext')
    computed_inv_sha1 = osutils.sha_string(bytes)
    self.assertEqual(computed_inv_sha1, recorded_inv_sha1)
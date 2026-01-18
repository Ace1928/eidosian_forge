from breezy import tests, urlutils
from breezy.bzr import remote
from breezy.tests.per_repository import TestCaseWithRepository
def test_doesnt_call_get_parent_map_on_all_fallback_revs(self):
    if not isinstance(self.repository_format, remote.RemoteRepositoryFormat):
        raise tests.TestNotApplicable('only for RemoteRepository')
    master_b, stacked_b = self.make_stacked_branch_with_long_history()
    self.addCleanup(stacked_b.lock_read().unlock)
    self.make_repository('target_repo', shared=True)
    target_b = self.make_branch('target_repo/branch')
    self.addCleanup(target_b.lock_write().unlock)
    self.setup_smart_server_with_call_log()
    res = target_b.repository.search_missing_revision_ids(stacked_b.repository, revision_ids=[b'F'], find_ghosts=False)
    self.assertParentMapCalls([(b'extra/stacked/', [b'F']), (b'extra/master/', [b'E']), (b'extra/target_repo/branch/', [b'A', b'B', b'C', b'D', b'E', b'F'])])
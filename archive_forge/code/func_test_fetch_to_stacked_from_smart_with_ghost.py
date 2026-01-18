from breezy import branch
from breezy.bzr import vf_search
from breezy.tests.per_repository import TestCaseWithRepository
def test_fetch_to_stacked_from_smart_with_ghost(self):
    source_b, base, stacked = self.make_source_with_ghost_and_stacked_target()
    trans = self.make_smart_server('source')
    source_b = branch.Branch.open(trans.base)
    source_b.lock_read()
    self.addCleanup(source_b.unlock)
    stacked.pull(source_b, stop_revision=b'B-id')
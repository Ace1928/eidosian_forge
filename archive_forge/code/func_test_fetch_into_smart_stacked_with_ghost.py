from breezy import branch
from breezy.bzr import vf_search
from breezy.tests.per_repository import TestCaseWithRepository
def test_fetch_into_smart_stacked_with_ghost(self):
    source_b, base, stacked = self.make_source_with_ghost_and_stacked_target()
    trans = self.make_smart_server('stacked')
    stacked = branch.Branch.open(trans.base)
    stacked.lock_write()
    self.addCleanup(stacked.unlock)
    stacked.pull(source_b, stop_revision=b'B-id')
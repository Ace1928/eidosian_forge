from breezy import branch
from breezy.bzr import vf_search
from breezy.tests.per_repository import TestCaseWithRepository
def test_fetch_with_ghost_stacked(self):
    source_b, base, stacked = self.make_source_with_ghost_and_stacked_target()
    stacked.pull(source_b, stop_revision=b'B-id')
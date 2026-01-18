from ..memorybranch import MemoryBranch
from . import TestCaseWithTransport
def test_get_rev_id(self):
    self.assertEqual(self.revid1, self.branch.get_rev_id(1))
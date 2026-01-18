from breezy import errors, osutils
from breezy.tests import features
from breezy.tests.per_lock import TestCaseWithLock
def test_multiple_read_locks(self):
    """You can take out more than one read lock on the same file."""
    a_lock = self.read_lock('a-file')
    self.addCleanup(a_lock.unlock)
    b_lock = self.read_lock('a-file')
    self.addCleanup(b_lock.unlock)
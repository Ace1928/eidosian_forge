from breezy import errors, osutils
from breezy.tests import features
from breezy.tests.per_lock import TestCaseWithLock
def test_multiple_write_locks_exclude(self):
    """Taking out more than one write lock should fail."""
    a_lock = self.write_lock('a-file')
    self.addCleanup(a_lock.unlock)
    self.assertRaises(errors.LockContention, self.write_lock, 'a-file')
from breezy import errors
from breezy.tests.per_lock import TestCaseWithLock
def test_is_write_locked(self):
    """With a temporary write lock, we cannot grab another lock."""
    a_lock = self.read_lock('a-file')
    try:
        success, t_write_lock = a_lock.temporary_write_lock()
        self.assertTrue(success, 'We failed to grab a write lock.')
        try:
            self.assertRaises(errors.LockContention, self.write_lock, 'a-file')
        finally:
            a_lock = t_write_lock.restore_read_lock()
        b_lock = self.read_lock('a-file')
        b_lock.unlock()
    finally:
        a_lock.unlock()
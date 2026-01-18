from breezy import errors
from breezy.tests.per_lock import TestCaseWithLock
def test_fails_when_locked(self):
    """We can't upgrade to a write lock if something else locks."""
    a_lock = self.read_lock('a-file')
    try:
        b_lock = self.read_lock('a-file')
        try:
            success, alt_lock = a_lock.temporary_write_lock()
            self.assertFalse(success)
            self.assertTrue(alt_lock is a_lock or a_lock.f is None)
            a_lock = alt_lock
            c_lock = self.read_lock('a-file')
            c_lock.unlock()
        finally:
            b_lock.unlock()
    finally:
        a_lock.unlock()
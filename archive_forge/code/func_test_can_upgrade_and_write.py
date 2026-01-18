from breezy import errors
from breezy.tests.per_lock import TestCaseWithLock
def test_can_upgrade_and_write(self):
    """With only one lock, we should be able to write lock and switch back."""
    a_lock = self.read_lock('a-file')
    try:
        success, t_write_lock = a_lock.temporary_write_lock()
        self.assertTrue(success, 'We failed to grab a write lock.')
        try:
            self.assertEqual(b'contents of a-file\n', t_write_lock.f.read())
            t_write_lock.f.seek(0)
            t_write_lock.f.write(b'new contents for a-file\n')
            t_write_lock.f.seek(0)
            self.assertEqual(b'new contents for a-file\n', t_write_lock.f.read())
        finally:
            a_lock = t_write_lock.restore_read_lock()
    finally:
        a_lock.unlock()
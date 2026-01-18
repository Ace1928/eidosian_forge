from breezy import errors, ui
from breezy.tests import per_repository
def test_locked(self):
    self.repo.lock_write()
    self.assertEqual(self.repo.get_physical_lock_status(), self.unused_repo.get_physical_lock_status())
    if not self.unused_repo.get_physical_lock_status():
        self.repo.unlock()
        return
    self.unused_repo.break_lock()
    self.assertRaises(errors.LockBroken, self.repo.unlock)
from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
def test_05_lock_read_fail_repo(self):
    b = self.get_instrumented_branch()
    b.repository.disable_lock_read()
    self.assertRaises(lock_helpers.TestPreventLocking, b.lock_read)
    self.assertFalse(b.is_locked())
    self.assertFalse(b.repository.is_locked())
    self.assertEqual([('b', 'lr', True), ('r', 'lr', False)], self.locks)
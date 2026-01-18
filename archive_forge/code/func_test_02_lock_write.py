from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
def test_02_lock_write(self):
    b = self.get_instrumented_branch()
    self.assertFalse(b.is_locked())
    self.assertFalse(b.repository.is_locked())
    b.lock_write()
    try:
        self.assertTrue(b.is_locked())
        self.assertTrue(b.repository.is_locked())
    finally:
        b.unlock()
    self.assertFalse(b.is_locked())
    self.assertFalse(b.repository.is_locked())
    if self.combined_control:
        self.assertEqual([('b', 'lw', True), ('r', 'lw', True), ('rc', 'lw', True), ('bc', 'lw', True), ('b', 'ul', True), ('bc', 'ul', True), ('r', 'ul', True), ('rc', 'ul', True)], self.locks)
    else:
        self.assertEqual([('b', 'lw', True), ('r', 'lw', True), ('bc', 'lw', True), ('b', 'ul', True), ('bc', 'ul', True), ('r', 'ul', True)], self.locks)
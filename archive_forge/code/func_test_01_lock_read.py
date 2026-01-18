from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
def test_01_lock_read(self):
    b = self.get_instrumented_branch()
    self.assertFalse(b.is_locked())
    self.assertFalse(b.repository.is_locked())
    b.lock_read()
    try:
        self.assertTrue(b.is_locked())
        self.assertTrue(b.repository.is_locked())
    finally:
        b.unlock()
    self.assertFalse(b.is_locked())
    self.assertFalse(b.repository.is_locked())
    if self.combined_control:
        self.assertEqual([('b', 'lr', True), ('r', 'lr', True), ('rc', 'lr', True), ('bc', 'lr', True), ('b', 'ul', True), ('bc', 'ul', True), ('r', 'ul', True), ('rc', 'ul', True)], self.locks)
    else:
        self.assertEqual([('b', 'lr', True), ('r', 'lr', True), ('bc', 'lr', True), ('b', 'ul', True), ('bc', 'ul', True), ('r', 'ul', True)], self.locks)
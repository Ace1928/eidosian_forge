from ..counted_lock import CountedLock
from ..errors import LockError, LockNotHeld, ReadOnlyError, TokenMismatch
from . import TestCase
def test_lock_initially_not_held(self):
    l = DummyLock()
    self.assertFalse(l.is_locked())
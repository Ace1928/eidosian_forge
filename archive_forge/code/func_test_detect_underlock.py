from ..counted_lock import CountedLock
from ..errors import LockError, LockNotHeld, ReadOnlyError, TokenMismatch
from . import TestCase
def test_detect_underlock(self):
    l = DummyLock()
    self.assertRaises(LockError, l.unlock)
from twisted.python import reflect, win32
from twisted.trial import unittest
def test_deprecation_WindowsError(self) -> None:
    """Importing C{WindowsError} triggers a L{DeprecationWarning}."""
    self.assertWarns(DeprecationWarning, "twisted.python.win32.WindowsError was deprecated in Twisted 21.2.0: Catch OSError and check presence of 'winerror' attribute.", reflect.__file__, lambda: reflect.namedAny('twisted.python.win32.WindowsError'))
import sys
from twisted.python.reflect import namedModule
from twisted.python.runtime import Platform, shortPythonVersion
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import suppress as SUPRESS
def test_getType(self) -> None:
    """
        If an operating system name is supplied to L{Platform}'s initializer,
        L{Platform.getType} returns the platform type which corresponds to that
        name.
        """
    self.assertEqual(Platform('nt').getType(), 'win32')
    self.assertEqual(Platform('ce').getType(), 'win32')
    self.assertEqual(Platform('posix').getType(), 'posix')
    self.assertEqual(Platform('java').getType(), 'java')
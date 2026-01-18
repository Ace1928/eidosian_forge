import sys
from twisted.python.reflect import namedModule
from twisted.python.runtime import Platform, shortPythonVersion
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import suppress as SUPRESS
def test_isMacOSXConsistency(self) -> None:
    """
        L{Platform.isMacOSX} can only return C{True} if L{Platform.getType}
        returns C{'posix'}.
        """
    platform = Platform()
    if platform.isMacOSX():
        self.assertEqual(platform.getType(), 'posix')
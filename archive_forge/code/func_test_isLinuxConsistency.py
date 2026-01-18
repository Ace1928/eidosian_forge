import sys
from twisted.python.reflect import namedModule
from twisted.python.runtime import Platform, shortPythonVersion
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import suppress as SUPRESS
def test_isLinuxConsistency(self) -> None:
    """
        L{Platform.isLinux} can only return C{True} if L{Platform.getType}
        returns C{'posix'} and L{sys.platform} starts with C{"linux"}.
        """
    platform = Platform()
    if platform.isLinux():
        self.assertTrue(sys.platform.startswith('linux'))
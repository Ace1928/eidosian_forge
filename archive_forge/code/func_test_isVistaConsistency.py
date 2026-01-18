import sys
from twisted.python.reflect import namedModule
from twisted.python.runtime import Platform, shortPythonVersion
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import suppress as SUPRESS
def test_isVistaConsistency(self) -> None:
    """
        Verify consistency of L{Platform.isVista}: it can only be C{True} if
        L{Platform.isWinNT} and L{Platform.isWindows} are C{True}.
        """
    platform = Platform()
    if platform.isVista():
        self.assertTrue(platform.isWinNT())
        self.assertTrue(platform.isWindows())
        self.assertFalse(platform.isMacOSX())
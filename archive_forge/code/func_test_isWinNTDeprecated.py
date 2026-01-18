import sys
from twisted.python.reflect import namedModule
from twisted.python.runtime import Platform, shortPythonVersion
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import suppress as SUPRESS
def test_isWinNTDeprecated(self) -> None:
    """
        L{Platform.isWinNT} is deprecated in favor of L{platform.isWindows}.
        """
    platform = Platform()
    platform.isWinNT()
    warnings = self.flushWarnings([self.test_isWinNTDeprecated])
    self.assertEqual(len(warnings), 1)
    self.assertEqual(warnings[0]['message'], self.isWinNTDeprecationMessage)
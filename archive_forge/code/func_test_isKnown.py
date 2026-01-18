import sys
from twisted.python.reflect import namedModule
from twisted.python.runtime import Platform, shortPythonVersion
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import suppress as SUPRESS
def test_isKnown(self) -> None:
    """
        L{Platform.isKnown} returns a boolean indicating whether this is one of
        the L{runtime.knownPlatforms}.
        """
    platform = Platform()
    self.assertTrue(platform.isKnown())
import sys
from twisted.python.reflect import namedModule
from twisted.python.runtime import Platform, shortPythonVersion
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import suppress as SUPRESS
def test_noCGroups(self) -> None:
    """
        If the platform is Linux, and the cgroups file in C{/proc} does not
        exist, C{isDocker()} returns L{False}
        """
    platform = Platform(None, 'linux')
    self.assertFalse(platform.isDocker(_initCGroupLocation='fakepath'))
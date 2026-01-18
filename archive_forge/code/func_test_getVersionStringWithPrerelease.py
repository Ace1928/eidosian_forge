import operator
from incremental import _inf
from twisted.python.versions import IncomparableVersions, Version, getVersionString
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_getVersionStringWithPrerelease(self) -> None:
    """
        L{getVersionString} includes the prerelease, if any.
        """
    self.assertEqual(getVersionString(Version('whatever', 8, 0, 0, prerelease=1)), 'whatever 8.0.0.rc1')
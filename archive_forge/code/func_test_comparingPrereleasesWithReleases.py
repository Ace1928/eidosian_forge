import operator
from incremental import _inf
from twisted.python.versions import IncomparableVersions, Version, getVersionString
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_comparingPrereleasesWithReleases(self) -> None:
    """
        Prereleases are always less than versions without prereleases.
        """
    va = Version('whatever', 1, 0, 0, prerelease=1)
    vb = Version('whatever', 1, 0, 0)
    self.assertTrue(va < vb)
    self.assertFalse(va > vb)
    self.assertNotEqual(vb, va)
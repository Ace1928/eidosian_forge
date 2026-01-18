import operator
from incremental import _inf
from twisted.python.versions import IncomparableVersions, Version, getVersionString
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_baseWithPrerelease(self) -> None:
    """
        The base version includes 'preX' for versions with prereleases.
        """
    self.assertEqual(Version('foo', 1, 0, 0, prerelease=8).base(), '1.0.0.rc8')
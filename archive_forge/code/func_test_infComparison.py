import operator
from incremental import _inf
from twisted.python.versions import IncomparableVersions, Version, getVersionString
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_infComparison(self) -> None:
    """
        L{_inf} is equal to L{_inf}.

        This is a regression test.
        """
    self.assertEqual(_inf, _inf)
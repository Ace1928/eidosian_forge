from twisted.trial.unittest import SynchronousTestCase
from .._convenience import Quit
from .._ithreads import AlreadyQuit
def test_isInitiallySet(self) -> None:
    """
        L{Quit.isSet} starts as L{False}.
        """
    quit = Quit()
    self.assertEqual(quit.isSet, False)
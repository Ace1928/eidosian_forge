from twisted.trial.unittest import SynchronousTestCase
from .._convenience import Quit
from .._ithreads import AlreadyQuit
def test_checkAfterSetRaises(self) -> None:
    """
        L{Quit.check} raises L{AlreadyQuit} if L{Quit.set} has been called.
        """
    quit = Quit()
    quit.set()
    self.assertRaises(AlreadyQuit, quit.check)
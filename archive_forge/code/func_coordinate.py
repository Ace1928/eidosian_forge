from twisted.python.components import proxyForInterface
from twisted.python.context import call, get
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, Team, createMemoryWorker
def coordinate(self):
    """
        Perform all work currently scheduled in the coordinator.

        @return: whether any coordination work was performed; if the
            coordinator was idle when this was called, return L{False}
            (otherwise L{True}).
        @rtype: L{bool}
        """
    did = False
    while self.coordinateOnce():
        did = True
    return did
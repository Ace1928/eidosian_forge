from twisted.internet import threads
from twisted.python import log, reflect
def finalClose(self):
    """
        This should only be called by the shutdown trigger.
        """
    self.shutdownID = None
    self.threadpool.stop()
    self.running = False
    for conn in self.connections.values():
        self._close(conn)
    self.connections.clear()
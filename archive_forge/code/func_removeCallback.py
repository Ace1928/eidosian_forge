from twisted.python import log
from twisted.words.xish import xpath
def removeCallback(self, method):
    """
        Remove callback.

        @param method: The callable to be removed.
        """
    if method in self.callbacks:
        del self.callbacks[method]
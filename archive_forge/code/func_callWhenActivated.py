import time
from twisted.internet import defer
from twisted.spread import banana, flavors, jelly
def callWhenActivated(self, callback):
    """Externally register for notification when this publishable has received all relevant data."""
    if self.isActivated:
        callback(self)
    else:
        self._activationListeners.append(callback)
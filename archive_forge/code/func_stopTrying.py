import random
from typing import Any, Callable, Optional
from zope.interface import implementer
from twisted.internet import defer, error, interfaces
from twisted.internet.interfaces import IAddress, ITransport
from twisted.logger import _loggerFor
from twisted.python import components, failure, log
def stopTrying(self):
    """
        Put a stop to any attempt to reconnect in progress.
        """
    if self._callID:
        self._callID.cancel()
        self._callID = None
    self.continueTrying = 0
    if self.connector:
        try:
            self.connector.stopConnecting()
        except error.NotConnectingError:
            pass
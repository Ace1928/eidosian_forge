import random
from typing import Any, Callable, Optional
from zope.interface import implementer
from twisted.internet import defer, error, interfaces
from twisted.internet.interfaces import IAddress, ITransport
from twisted.logger import _loggerFor
from twisted.python import components, failure, log
def resetDelay(self):
    """
        Call this method after a successful connection: it resets the delay and
        the retry counter.
        """
    self.delay = self.initialDelay
    self.retries = 0
    self._callID = None
    self.continueTrying = 1
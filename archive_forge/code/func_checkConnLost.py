import msvcrt
import os
from zope.interface import implementer
import win32api
from twisted.internet import _pollingfile, main
from twisted.internet.interfaces import (
from twisted.python.failure import Failure
def checkConnLost(self):
    self.connsLost += 1
    if self.connsLost >= 2:
        self.disconnecting = True
        self.disconnected = True
        self.proto.connectionLost(Failure(main.CONNECTION_DONE))
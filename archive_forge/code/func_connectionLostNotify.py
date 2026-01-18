import os
import sys
from zope.interface import implementer
import pywintypes
import win32api
import win32con
import win32event
import win32file
import win32pipe
import win32process
import win32security
from twisted.internet import _pollingfile, error
from twisted.internet._baseprocess import BaseProcess
from twisted.internet.interfaces import IConsumer, IProcessTransport, IProducer
from twisted.python.win32 import quoteArguments
def connectionLostNotify(self):
    """
        Will be called 3 times, by stdout/err threads and process handle.
        """
    self.closedNotifies += 1
    self.maybeCallProcessEnded()
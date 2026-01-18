import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
def registerWritten(self, length):
    """
        Called by protocol to tell us more bytes were written.
        """
    self.writtenThisSecond += length
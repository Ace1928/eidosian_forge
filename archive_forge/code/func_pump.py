import gc
import os
import sys
import time
import weakref
from collections import deque
from io import BytesIO as StringIO
from typing import Dict
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import address, main, protocol, reactor
from twisted.internet.defer import Deferred, gatherResults, succeed
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.testing import _FakeConnector
from twisted.protocols.policies import WrappingFactory
from twisted.python import failure, log
from twisted.python.compat import iterbytes
from twisted.spread import jelly, pb, publish, util
from twisted.trial import unittest
def pump(self):
    """
        Move data back and forth.

        Returns whether any data was moved.
        """
    self.clientIO.seek(0)
    self.serverIO.seek(0)
    cData = self.clientIO.read()
    sData = self.serverIO.read()
    self.clientIO.seek(0)
    self.serverIO.seek(0)
    self.clientIO.truncate()
    self.serverIO.truncate()
    self.client.transport._checkProducer()
    self.server.transport._checkProducer()
    for byte in iterbytes(cData):
        self.server.dataReceived(byte)
    for byte in iterbytes(sData):
        self.client.dataReceived(byte)
    if cData or sData:
        return 1
    else:
        return 0
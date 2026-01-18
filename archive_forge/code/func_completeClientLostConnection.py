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
def completeClientLostConnection(self, reason=failure.Failure(main.CONNECTION_DONE)):
    """
        Asserts that the client broker's transport was closed and then
        mimics the event loop by calling the broker's connectionLost
        callback with C{reason}, followed by C{self.clientFactory}'s
        C{clientConnectionLost}

        @param reason: (optional) the reason to pass to the client
            broker's connectionLost callback
        @type reason: L{Failure}
        """
    self.assertTrue(self.client.transport.closed)
    self.client.connectionLost(reason)
    self.clientFactory.clientConnectionLost(self.connector, reason)
import base64
import calendar
import random
from io import BytesIO
from itertools import cycle
from typing import Sequence, Union
from unittest import skipIf
from urllib.parse import clear_cache  # type: ignore[attr-defined]
from urllib.parse import urlparse, urlunsplit
from zope.interface import directlyProvides, providedBy, provider
from zope.interface.verify import verifyObject
import hamcrest
from twisted.internet import address
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols import loopback
from twisted.python.compat import iterbytes, networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
from twisted.web import http, http_headers, iweb
from twisted.web.http import PotentialDataLoss, _DataLoss, _IdentityTransferDecoder
from twisted.web.test.requesthelper import (
from ._util import assertIsFilesystemTemporary
def runRequest(self, httpRequest, requestFactory=None, success=True, channel=None):
    """
        Execute a web request based on plain text content.

        @param httpRequest: Content for the request which is processed. Each
            L{"
"} will be replaced with L{"\r
"}.
        @type httpRequest: C{bytes}

        @param requestFactory: 2-argument callable returning a Request.
        @type requestFactory: C{callable}

        @param success: Value to compare against I{self.didRequest}.
        @type success: C{bool}

        @param channel: Channel instance over which the request is processed.
        @type channel: L{HTTPChannel}

        @return: Returns the channel used for processing the request.
        @rtype: L{HTTPChannel}
        """
    if not channel:
        channel = http.HTTPChannel()
    if requestFactory:
        channel.requestFactory = _makeRequestProxyFactory(requestFactory)
    httpRequest = httpRequest.replace(b'\n', b'\r\n')
    transport = StringTransport()
    channel.makeConnection(transport)
    for byte in iterbytes(httpRequest):
        if channel.transport.disconnecting:
            break
        channel.dataReceived(byte)
    channel.connectionLost(IOError('all done'))
    if success:
        self.assertTrue(self.didRequest)
    else:
        self.assertFalse(self.didRequest)
    return channel
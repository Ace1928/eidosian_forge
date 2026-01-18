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
def runChunkedRequest(self, httpRequest, requestFactory=None, chunkSize=1):
    """
        Execute a web request based on plain text content, chunking
        the request payload.

        This is a stripped-down, chunking version of ParsingTests.runRequest.
        """
    channel = http.HTTPChannel()
    if requestFactory:
        channel.requestFactory = _makeRequestProxyFactory(requestFactory)
    httpRequest = httpRequest.replace(b'\n', b'\r\n')
    header, body = httpRequest.split(b'\r\n\r\n', 1)
    transport = StringTransport()
    channel.makeConnection(transport)
    channel.dataReceived(header + b'\r\n\r\n')
    for pos in range(len(body) // chunkSize + 1):
        if channel.transport.disconnecting:
            break
        channel.dataReceived(b''.join(http.toChunk(body[pos * chunkSize:(pos + 1) * chunkSize])))
    channel.dataReceived(b''.join(http.toChunk(b'')))
    channel.connectionLost(IOError('all done'))
    return channel
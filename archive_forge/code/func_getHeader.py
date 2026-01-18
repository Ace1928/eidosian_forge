from __future__ import annotations
from io import BytesIO
from typing import Dict, List, Optional
from zope.interface import implementer, verify
from incremental import Version
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, ISSLTransport
from twisted.internet.task import Clock
from twisted.python.deprecate import deprecated
from twisted.trial import unittest
from twisted.web._responses import FOUND
from twisted.web.http_headers import Headers
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET, Session, Site
def getHeader(self, name):
    """
        Retrieve the value of a request header.

        @type name: C{bytes}
        @param name: The name of the request header for which to retrieve the
            value.  Header names are compared case-insensitively.

        @rtype: C{bytes} or L{None}
        @return: The value of the specified request header.
        """
    return self.requestHeaders.getRawHeaders(name.lower(), [None])[0]
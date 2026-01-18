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
def endRequest(self):
    pass
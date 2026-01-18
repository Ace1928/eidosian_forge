import os
import zlib
from io import BytesIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet import interfaces
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.task import Clock
from twisted.internet.testing import EventLoggingObserver, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python import failure, reflect
from twisted.python.compat import iterbytes
from twisted.python.filepath import FilePath
from twisted.trial import unittest
from twisted.web import error, http, iweb, resource, server
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET, Request, Site
from twisted.web.static import Data
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from ._util import assertIsFilesystemTemporary
def test_computeAllowedMethods(self):
    """
        C{_computeAllowedMethods} will search through the
        'gettableresource' for all attributes/methods of the form
        'render_{method}' ('render_GET', for example) and return a list of
        the methods. 'HEAD' will always be included from the
        resource.Resource superclass.
        """
    res = GettableResource()
    allowedMethods = resource._computeAllowedMethods(res)
    self.assertEqual(set(allowedMethods), {b'GET', b'HEAD', b'fred_render_ethel'})
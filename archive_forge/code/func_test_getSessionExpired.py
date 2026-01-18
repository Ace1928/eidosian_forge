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
def test_getSessionExpired(self):
    """
        L{Request.getSession} generates a new session when the previous
        session has expired.
        """
    clock = Clock()
    site = server.Site(resource.Resource())
    d = DummyChannel()
    request = server.Request(d, 1)
    request.site = site
    request.sitepath = []

    def sessionFactoryWithClock(site, uid):
        """
            Forward to normal session factory, but inject the clock.

            @param site: The site on which the session is created.
            @type site: L{server.Site}

            @param uid: A unique identifier for the session.
            @type uid: C{bytes}

            @return: A newly created session.
            @rtype: L{server.Session}
            """
        session = sessionFactory(site, uid)
        session._reactor = clock
        return session
    sessionFactory = site.sessionFactory
    site.sessionFactory = sessionFactoryWithClock
    initialSession = request.getSession()
    clock.advance(sessionFactory.sessionTimeout)
    newSession = request.getSession()
    self.addCleanup(newSession.expire)
    self.assertIsNot(initialSession, newSession)
    self.assertNotEqual(initialSession.uid, newSession.uid)
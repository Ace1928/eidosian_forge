import tempfile
import traceback
import warnings
from sys import exc_info
from urllib.parse import quote as urlquote
from zope.interface.verify import verifyObject
from twisted.internet import reactor
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.error import ConnectionLost
from twisted.internet.testing import EventLoggingObserver
from twisted.logger import Logger, globalLogPublisher
from twisted.python.failure import Failure
from twisted.python.threadable import getThreadID
from twisted.python.threadpool import ThreadPool
from twisted.trial.unittest import TestCase
from twisted.web import http
from twisted.web.resource import IResource, Resource
from twisted.web.server import Request, Site, version
from twisted.web.test.test_web import DummyChannel
from twisted.web.wsgi import WSGIResource
def test_multipleStartResponse(self):
    """
        If the I{start_response} callable is invoked multiple times before a
        data for the response body is produced, the values from the last call
        are used.
        """
    channel = DummyChannel()

    def applicationFactory():

        def application(environ, startResponse):
            startResponse('100 Foo', [])
            startResponse('200 Bar', [])
            return iter(())
        return application
    d, requestFactory = self.requestFactoryFactory()

    def cbRendered(ignored):
        self.assertTrue(channel.transport.written.getvalue().startswith(b'HTTP/1.1 200 Bar\r\n'))
    d.addCallback(cbRendered)
    self.lowLevelRender(requestFactory, applicationFactory, lambda: channel, 'GET', '1.1', [], [''], None, [])
    return d
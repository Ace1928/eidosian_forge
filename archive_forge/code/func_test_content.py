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
def test_content(self):
    """
        Content produced by the iterator returned by the application object is
        written to the request as it is produced.
        """
    channel = DummyChannel()
    intermediateValues = []

    def record():
        intermediateValues.append(channel.transport.written.getvalue())

    def applicationFactory():

        def application(environ, startResponse):
            startResponse('200 OK', [('content-length', '6')])
            yield b'foo'
            record()
            yield b'bar'
            record()
        return application
    d, requestFactory = self.requestFactoryFactory()

    def cbRendered(ignored):
        self.assertEqual(self.getContentFromResponse(intermediateValues[0]), b'foo')
        self.assertEqual(self.getContentFromResponse(intermediateValues[1]), b'foobar')
    d.addCallback(cbRendered)
    self.lowLevelRender(requestFactory, applicationFactory, lambda: channel, 'GET', '1.1', [], [''], None, [])
    return d
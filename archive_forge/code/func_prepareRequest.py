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
def prepareRequest(self, application=None):
    """
        Prepare a L{Request} which, when a request is received, captures the
        C{environ} and C{start_response} callable passed to a WSGI app.

        @param application: An optional WSGI application callable that accepts
            the familiar C{environ} and C{start_response} args and returns an
            iterable of body content. If not supplied, C{start_response} will
            be called with a "200 OK" status and no headers, and no content
            will be yielded.

        @return: A two-tuple of (C{request}, C{deferred}). The former is a
            Twisted L{Request}. The latter is a L{Deferred} which will be
            called back with a two-tuple of the arguments passed to a WSGI
            application (i.e. the C{environ} and C{start_response} callable),
            or will errback with any error arising within the WSGI app.
        """
    result = Deferred()

    def outerApplication(environ, startResponse):
        try:
            if application is None:
                startResponse('200 OK', [])
                content = iter(())
            else:
                content = application(environ, startResponse)
        except BaseException:
            result.errback()
            startResponse('500 Error', [])
            return iter(())
        else:
            result.callback((environ, startResponse))
            return content
    resource = WSGIResource(self.reactor, self.threadpool, outerApplication)
    root = Resource()
    root.putChild(b'res', resource)
    channel = self.channelFactory()
    channel.site = Site(root)

    class CannedRequest(Request):
        """
            Convenient L{Request} derivative which has canned values for all
            of C{requestReceived}'s arguments.
            """

        def requestReceived(self, command=b'GET', path=b'/res', version=b'1.1'):
            return Request.requestReceived(self, command=command, path=path, version=version)
    request = CannedRequest(channel, queued=False)
    request.gotLength(0)
    return (request, result)
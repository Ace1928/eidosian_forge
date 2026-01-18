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
def test_contentTypeIsNativeString(self):
    """
        The C{'CONTENT_TYPE'} key of the C{environ} C{dict} passed to the
        application is always a native string.
        """
    for contentType in (b'x-foo/bar', 'x-foo/bar'):
        request, result = self.prepareRequest()
        request.requestHeaders.addRawHeader(b'Content-Type', contentType)
        request.requestReceived()
        result.addCallback(self.environKeyEqual('CONTENT_TYPE', 'x-foo/bar'))
        self.assertIsInstance(self.successResultOf(result), str)
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
def test_environIsDict(self):
    """
        L{WSGIResource} calls the application object with an C{environ}
        parameter which is exactly of type C{dict}.
        """
    d = self.render('GET', '1.1', [], [''])

    def cbRendered(result):
        environ, startResponse = result
        self.assertIdentical(type(environ), dict)
        for name in environ:
            self.assertIsInstance(name, str)
    d.addCallback(cbRendered)
    return d
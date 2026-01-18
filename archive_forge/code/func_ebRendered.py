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
def ebRendered(ignored):
    self.assertEquals(1, len(logObserver))
    event = logObserver[0]
    f = event['log_failure']
    self.assertIsInstance(f.value, RuntimeError)
    self.flushLoggedErrors(RuntimeError)
    response = channel.transport.written.getvalue()
    self.assertTrue(response.startswith(b'HTTP/1.1 200 OK'))
    self.assertIn(responseContent, response)
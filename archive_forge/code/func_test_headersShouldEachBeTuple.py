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
def test_headersShouldEachBeTuple(self):
    """
        According to PEP-3333, each header passed to the I{start_response}
        callable should be a tuple:

          The response_headers argument is a list of (header_name,
          header_value) tuples

        However, for bug-compatibility, any 2 element sequence is also
        accepted. In both Python 2 and Python 3, only a warning is issued when
        a sequence other than a tuple is encountered.
        """

    def application(environ, startResponse):
        startResponse('200 OK', [['not', 'tuple']])
        return iter(())
    request, result = self.prepareRequest(application)
    with warnings.catch_warnings(record=True) as caught:
        request.requestReceived()
        result = self.successResultOf(result)
    self.assertEqual(1, len(caught))
    self.assertEqual(RuntimeWarning, caught[0].category)
    self.assertEqual("header should be a (str, str) tuple, not ['not', 'tuple'] (list)", str(caught[0].message))
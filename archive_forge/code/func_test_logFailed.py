from os.path import abspath
from xml.dom.minidom import parseString
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import defer, reactor
from twisted.logger import globalLogPublisher
from twisted.python import failure, filepath
from twisted.spread import pb
from twisted.spread.banana import SIZE_LIMIT
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
from twisted.web import client, distrib, resource, server, static
from twisted.web.http_headers import Headers
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
def test_logFailed(self):
    """
        When a request fails, the string form of the failure is logged.
        """
    logObserver = proto_helpers.EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
    f = failure.Failure(ArbitraryError())
    request = DummyRequest([b''])
    issue = distrib.Issue(request)
    issue.failed(f)
    self.assertEquals(1, len(logObserver))
    self.assertIn('Failure instance', logObserver[0]['log_format'])
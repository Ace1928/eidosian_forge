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
def test_largeReturn(self):
    """
        Like L{test_largeWrite}, but for the case where C{render} returns a
        long string rather than explicitly passing it to L{Request.write}.
        """

    class LargeReturn(resource.Resource):

        def render(self, request):
            return b'x' * SIZE_LIMIT + b'y'
    request = self._requestTest(LargeReturn())
    request.addCallback(self.assertEqual, b'x' * SIZE_LIMIT + b'y')
    return request
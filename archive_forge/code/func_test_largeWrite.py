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
def test_largeWrite(self):
    """
        If a string longer than the Banana size limit is passed to the
        L{distrib.Request} passed to the remote resource, it is broken into
        smaller strings to be transported over the PB connection.
        """

    class LargeWrite(resource.Resource):

        def render(self, request):
            request.write(b'x' * SIZE_LIMIT + b'y')
            request.finish()
            return server.NOT_DONE_YET
    request = self._requestTest(LargeWrite())
    request.addCallback(self.assertEqual, b'x' * SIZE_LIMIT + b'y')
    return request
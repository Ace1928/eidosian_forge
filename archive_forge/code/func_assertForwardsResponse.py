from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
def assertForwardsResponse(self, request, code, message, headers, body):
    """
        Assert that C{request} has forwarded a response from the server.

        @param request: A L{DummyRequest}.
        @param code: The expected HTTP response code.
        @param message: The expected HTTP message.
        @param headers: The expected HTTP headers.
        @param body: The expected response body.
        """
    self.assertEqual(request.responseCode, code)
    self.assertEqual(request.responseMessage, message)
    receivedHeaders = list(request.responseHeaders.getAllRawHeaders())
    receivedHeaders.sort()
    expectedHeaders = headers[:]
    expectedHeaders.sort()
    self.assertEqual(receivedHeaders, expectedHeaders)
    self.assertEqual(b''.join(request.written), body)
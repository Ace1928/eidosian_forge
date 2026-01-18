from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
def makeProxyClient(self, request, method=b'GET', headers=None, requestBody=b''):
    """
        Make a L{ProxyClient} object used for testing.

        @param request: The request to use.
        @param method: The HTTP method to use, GET by default.
        @param headers: The HTTP headers to use expressed as a dict. If not
            provided, defaults to {'accept': 'text/html'}.
        @param requestBody: The body of the request. Defaults to the empty
            string.
        @return: A L{ProxyClient}
        """
    if headers is None:
        headers = {b'accept': b'text/html'}
    path = b'/' + request.postpath
    return ProxyClient(method, path, b'HTTP/1.0', headers, requestBody, request)
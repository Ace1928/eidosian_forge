from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
def test_keepaliveNotForwarded(self):
    """
        The proxy doesn't really know what to do with keepalive things from
        the remote server, so we stomp over any keepalive header we get from
        the client.
        """
    headers = {b'accept': b'text/html', b'keep-alive': b'300', b'connection': b'keep-alive'}
    expectedHeaders = headers.copy()
    expectedHeaders[b'connection'] = b'close'
    del expectedHeaders[b'keep-alive']
    client = ProxyClient(b'GET', b'/foo', b'HTTP/1.0', headers, b'', None)
    self.assertForwardsHeaders(client, b'GET /foo HTTP/1.0', expectedHeaders)
from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
def test_losesConnection(self):
    """
        If the response contains a I{Content-Length} header, the outgoing
        connection is closed when all response body data has been received.
        """
    data = b'foo bar baz'
    return self._testDataForward(200, b'OK', [(b'Content-Length', [str(len(data)).encode('ascii')])], data, loseConnection=False)
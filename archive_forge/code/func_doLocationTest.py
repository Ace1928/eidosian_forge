import gc
from twisted.internet import defer
from twisted.python.compat import networkString
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase, TestCase
from twisted.web import resource, util
from twisted.web.error import FlattenerError
from twisted.web.http import FOUND
from twisted.web.server import Request
from twisted.web.template import TagLoader, flattenString, tags
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from twisted.web.util import (
def doLocationTest(self, requestPath: bytes) -> bytes:
    """
        Render a response to a request with path *requestPath*

        @param requestPath: A slash-separated path like C{b'/foo/bar'}.

        @returns: The value of the I{Location} header.
        """
    request = Request(DummyChannel(), True)
    request.method = b'GET'
    request.prepath = requestPath.lstrip(b'/').split(b'/')
    resource = ParentRedirect()
    resource.render(request)
    headers = request.responseHeaders.getRawHeaders(b'Location')
    assert headers is not None
    [location] = headers
    return location
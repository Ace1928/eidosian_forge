from typing import cast
from twisted.trial.unittest import SynchronousTestCase
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.pages import errorPage, forbidden, notFound
from twisted.web.resource import IResource
from twisted.web.test.requesthelper import DummyRequest
def assertResponse(self, request: DummyRequest, code: int, body: bytes) -> None:
    self.assertEqual(request.responseCode, code)
    self.assertEqual(request.responseHeaders, Headers({b'content-type': [b'text/html; charset=utf-8']}))
    self.assertEqual(b''.join(request.written).decode('latin-1'), body.decode('latin-1'))
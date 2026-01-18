from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
def test_noResourceRendering(self) -> None:
    """
        L{NoResource} sets the HTTP I{NOT FOUND} code.
        """
    detail = 'long message'
    page = self.noResource(detail)
    self._pageRenderingTest(page, NOT_FOUND, 'No Such Resource', detail)
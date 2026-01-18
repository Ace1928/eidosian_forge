from typing import cast
from twisted.trial.unittest import SynchronousTestCase
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.pages import errorPage, forbidden, notFound
from twisted.web.resource import IResource
from twisted.web.test.requesthelper import DummyRequest
def test_forbiddenDefaults(self) -> None:
    """
        The default arguments to L{twisted.web.pages.forbidden} produce
        a reasonable error page.
        """
    self.assertResponse(_render(forbidden()), 403, b'<!DOCTYPE html>\n<html><head><title>403 - Forbidden Resource</title></head><body><h1>Forbidden Resource</h1><p>Sorry, resource is forbidden.</p></body></html>')
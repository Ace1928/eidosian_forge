from typing import cast
from twisted.trial.unittest import SynchronousTestCase
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.pages import errorPage, forbidden, notFound
from twisted.web.resource import IResource
from twisted.web.test.requesthelper import DummyRequest
def test_escapesHTML(self) -> None:
    """
        The I{brief} and I{detail} parameters are HTML-escaped on render.
        """
    self.assertResponse(_render(errorPage(400, 'A & B', "<script>alert('oops!')")), 400, b"<!DOCTYPE html>\n<html><head><title>400 - A &amp; B</title></head><body><h1>A &amp; B</h1><p>&lt;script&gt;alert('oops!')</p></body></html>")
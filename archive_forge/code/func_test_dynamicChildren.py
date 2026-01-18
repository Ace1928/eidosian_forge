from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
def test_dynamicChildren(self) -> None:
    """
        L{Resource.getChildWithDefault} delegates to L{Resource.getChild} when
        the requested path is not associated with any static child.
        """
    path = b'foo'
    request = DummyRequest([])
    resource = DynamicChildren()
    child = resource.getChildWithDefault(path, request)
    self.assertIsInstance(child, DynamicChild)
    self.assertEqual(child.path, path)
    self.assertIdentical(child.request, request)
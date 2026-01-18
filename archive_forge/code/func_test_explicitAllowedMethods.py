from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
def test_explicitAllowedMethods(self) -> None:
    """
        The L{UnsupportedMethod} raised by L{Resource.render} for an unsupported
        request method has a C{allowedMethods} attribute set to the value of the
        C{allowedMethods} attribute of the L{Resource}, if it has one.
        """
    expected = [b'GET', b'HEAD', b'PUT']
    resource = Resource()
    resource.allowedMethods = expected
    request = DummyRequest([])
    request.method = b'FICTIONAL'
    exc = self.assertRaises(UnsupportedMethod, resource.render, request)
    self.assertEqual(set(expected), set(exc.allowedMethods))
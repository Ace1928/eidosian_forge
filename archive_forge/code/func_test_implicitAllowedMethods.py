from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
def test_implicitAllowedMethods(self) -> None:
    """
        The L{UnsupportedMethod} raised by L{Resource.render} for an unsupported
        request method has a C{allowedMethods} attribute set to a list of the
        methods supported by the L{Resource}, as determined by the
        I{render_}-prefixed methods which it defines, if C{allowedMethods} is
        not explicitly defined by the L{Resource}.
        """
    expected = {b'GET', b'HEAD', b'PUT'}
    resource = ImplicitAllowedMethods()
    request = DummyRequest([])
    request.method = b'FICTIONAL'
    exc = self.assertRaises(UnsupportedMethod, resource.render, request)
    self.assertEqual(expected, set(exc.allowedMethods))
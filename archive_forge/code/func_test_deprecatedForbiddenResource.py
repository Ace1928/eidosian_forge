from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
def test_deprecatedForbiddenResource(self) -> None:
    """
        The public C{twisted.web.resource.ForbiddenResource} alias for the
        corresponding C{_Unsafe} class produce a deprecation warning when
        imported.
        """
    from twisted.web.resource import ForbiddenResource
    self.assertIs(ForbiddenResource, self.forbiddenResource)
    [warning] = self.flushWarnings()
    self.assertEqual(warning['category'], DeprecationWarning)
    self.assertIn('twisted.web.pages.forbidden', warning['message'])
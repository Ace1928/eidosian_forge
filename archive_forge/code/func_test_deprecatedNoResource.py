from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
def test_deprecatedNoResource(self) -> None:
    """
        The public C{twisted.web.resource.NoResource} alias for the
        corresponding C{_Unsafe} class produces a deprecation warning when
        imported.
        """
    from twisted.web.resource import NoResource
    self.assertIs(NoResource, self.noResource)
    [warning] = self.flushWarnings()
    self.assertEqual(warning['category'], DeprecationWarning)
    self.assertIn('twisted.web.pages.notFound', warning['message'])
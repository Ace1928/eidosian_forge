import webob
from oslo_middleware.base import ConfigurableMiddleware
from oslo_middleware.base import Middleware
from oslotest.base import BaseTestCase
def test_extend_with_request(self):
    """Assert that a newer middleware behaves as appropriate.

        This tests makes sure that the request is passed to the
        middleware's implementation.
        """
    self.application = RequestBase(application)
    request = webob.Request({}, method='GET')
    request.get_response(self.application)
    self.assertTrue(self.application.called_with_request)
from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def assertCORSResponse(self, response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None, vary='Origin', has_content_type=False):
    """Test helper for CORS response headers.

        Assert all the headers in a given response. By default, we assume
        the response is empty.
        """
    self.assertEqual(status, response.status)
    self.assertHeader(response, 'Access-Control-Allow-Origin', allow_origin)
    self.assertHeader(response, 'Access-Control-Max-Age', max_age)
    self.assertHeader(response, 'Access-Control-Allow-Methods', allow_methods)
    self.assertHeader(response, 'Access-Control-Allow-Headers', allow_headers)
    self.assertHeader(response, 'Access-Control-Allow-Credentials', allow_credentials)
    self.assertHeader(response, 'Access-Control-Expose-Headers', expose_headers)
    if not has_content_type:
        self.assertHeader(response, 'Content-Type')
    if allow_origin:
        self.assertHeader(response, 'Vary', vary)
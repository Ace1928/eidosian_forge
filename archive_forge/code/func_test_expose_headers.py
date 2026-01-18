from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_expose_headers(self):
    """CORS Specification Section 6.1.4

        If the list of exposed headers is not empty add one or more
        Access-Control-Expose-Headers headers, with as values the header field
        names given in the list of exposed headers.
        """
    for method in self.methods:
        request = webob.Request.blank('/')
        request.method = method
        request.headers['Origin'] = 'http://headers.example.com'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin='http://headers.example.com', max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers='X-Header-1,X-Header-2', has_content_type=True)
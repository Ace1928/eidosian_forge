from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_no_request_headers(self):
    """CORS Specification Section 6.2.4

        If there are no Access-Control-Request-Headers headers let header
        field-names be the empty list.
        """
    request = webob.Request.blank('/')
    request.method = 'OPTIONS'
    request.headers['Origin'] = 'http://headers.example.com'
    request.headers['Access-Control-Request-Method'] = 'GET'
    request.headers['Access-Control-Request-Headers'] = ''
    response = request.get_response(self.application)
    self.assertCORSResponse(response, status='200 OK', allow_origin='http://headers.example.com', max_age=None, allow_methods='GET', allow_headers=None, allow_credentials=None, expose_headers=None)
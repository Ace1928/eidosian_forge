from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_no_request_method(self):
    """CORS Specification Section 6.2.3

        If there is no Access-Control-Request-Method header or if parsing
        failed, do not set any additional headers and terminate this set of
        steps. The request is outside the scope of this specification.
        """
    request = webob.Request.blank('/')
    request.method = 'OPTIONS'
    request.headers['Origin'] = 'http://get.example.com'
    request.headers['Access-Control-Request-Method'] = 'GET'
    response = request.get_response(self.application)
    self.assertCORSResponse(response, status='200 OK', allow_origin='http://get.example.com', max_age=None, allow_methods='GET', allow_headers=None, allow_credentials=None, expose_headers=None)
    request = webob.Request.blank('/')
    request.method = 'OPTIONS'
    request.headers['Origin'] = 'http://valid.example.com'
    request.headers['Access-Control-Request-Method'] = 'TEAPOT'
    response = request.get_response(self.application)
    self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None)
    request = webob.Request.blank('/')
    request.method = 'OPTIONS'
    request.headers['Origin'] = 'http://valid.example.com'
    response = request.get_response(self.application)
    self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None)
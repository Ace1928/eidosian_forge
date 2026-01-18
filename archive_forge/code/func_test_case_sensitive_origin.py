from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_case_sensitive_origin(self):
    """CORS Specification Section 6.2.2

        If the value of the Origin header is not a case-sensitive match for
        any of the values in list of origins do not set any additional headers
        and terminate this set of steps.
        """
    request = webob.Request.blank('/')
    request.method = 'OPTIONS'
    request.headers['Origin'] = 'http://valid.example.com'
    request.headers['Access-Control-Request-Method'] = 'GET'
    response = request.get_response(self.application)
    self.assertCORSResponse(response, status='200 OK', allow_origin='http://valid.example.com', max_age=None, allow_methods='GET', allow_headers='', allow_credentials=None, expose_headers=None)
    request = webob.Request.blank('/')
    request.method = 'OPTIONS'
    request.headers['Origin'] = 'http://invalid.example.com'
    request.headers['Access-Control-Request-Method'] = 'GET'
    response = request.get_response(self.application)
    self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None)
    request = webob.Request.blank('/')
    request.method = 'OPTIONS'
    request.headers['Origin'] = 'http://VALID.EXAMPLE.COM'
    request.headers['Access-Control-Request-Method'] = 'GET'
    response = request.get_response(self.application)
    self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None)
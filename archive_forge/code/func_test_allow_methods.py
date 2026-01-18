from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_allow_methods(self):
    """CORS Specification Section 6.2.9

        Add one or more Access-Control-Allow-Methods headers consisting of
        (a subset of) the list of methods.

        Since the list of methods can be unbounded, simply returning the method
        indicated by Access-Control-Request-Method (if supported) can be
        enough.
        """
    for method in ['GET', 'PUT', 'POST', 'DELETE']:
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://all.example.com'
        request.headers['Access-Control-Request-Method'] = method
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin='http://all.example.com', max_age=None, allow_methods=method, allow_headers=None, allow_credentials=None, expose_headers=None)
    for method in ['PUT', 'POST', 'DELETE']:
        request = webob.Request.blank('/')
        request.method = 'OPTIONS'
        request.headers['Origin'] = 'http://get.example.com'
        request.headers['Access-Control-Request-Method'] = method
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None)
from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_application_options_response(self):
    """Assert that an application provided OPTIONS response is honored.

        If the underlying application, via middleware or other, provides a
        CORS response, its response should be honored.
        """
    test_origin = 'http://creds.example.com'
    request = webob.Request.blank('/server_cors')
    request.method = 'OPTIONS'
    request.headers['Origin'] = test_origin
    request.headers['Access-Control-Request-Method'] = 'GET'
    response = request.get_response(self.application)
    self.assertNotIn('Access-Control-Allow-Credentials', response.headers)
    self.assertEqual(test_origin, response.headers['Access-Control-Allow-Origin'])
    self.assertEqual('1', response.headers['X-Server-Generated-Response'])
    request = webob.Request.blank('/server_no_cors')
    request.method = 'OPTIONS'
    request.headers['Origin'] = 'http://get.example.com'
    request.headers['Access-Control-Request-Method'] = 'GET'
    response = request.get_response(self.application)
    self.assertCORSResponse(response, status='200 OK', allow_origin='http://get.example.com', max_age=None, allow_methods='GET', allow_headers=None, allow_credentials=None, expose_headers=None, has_content_type=True)
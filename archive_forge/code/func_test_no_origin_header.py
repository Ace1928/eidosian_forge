from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_no_origin_header(self):
    """CORS Specification Section 6.2.1

        If the Origin header is not present terminate this set of steps. The
        request is outside the scope of this specification.
        """
    request = webob.Request.blank('/')
    request.method = 'OPTIONS'
    response = request.get_response(self.application)
    self.assertCORSResponse(response, status='200 OK', allow_origin=None, max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None)
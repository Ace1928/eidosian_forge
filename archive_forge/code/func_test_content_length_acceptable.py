import io
from oslo_config import fixture as config
from oslotest import base as test_base
import webob
from oslo_middleware import sizelimit
def test_content_length_acceptable(self):
    self.request.headers['Content-Length'] = self.MAX_REQUEST_BODY_SIZE
    self.request.body = b'0' * self.MAX_REQUEST_BODY_SIZE
    response = self.request.get_response(self.middleware)
    self.assertEqual(200, response.status_int)
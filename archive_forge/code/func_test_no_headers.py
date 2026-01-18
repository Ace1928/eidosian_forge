from wsgiref import util
from oslotest import base as test_base
import webob
from oslo_middleware import http_proxy_to_wsgi
def test_no_headers(self):
    response = self.request.get_response(self.middleware)
    self.assertEqual(b'http://localhost:80/', response.body)
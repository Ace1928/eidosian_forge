from wsgiref import util
from oslotest import base as test_base
import webob
from oslo_middleware import http_proxy_to_wsgi
def test__parse_rfc7239_header(self):
    expected_result = [{'for': 'foobar', 'proto': 'https'}, {'for': 'foobaz', 'proto': 'http'}]
    result = self.middleware._parse_rfc7239_header('for=foobar;proto=https, for=foobaz;proto=http')
    self.assertEqual(expected_result, result)
    result = self.middleware._parse_rfc7239_header('for=foobar; proto=https, for=foobaz; proto=http')
    self.assertEqual(expected_result, result)
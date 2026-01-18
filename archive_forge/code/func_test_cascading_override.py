from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_cascading_override(self):
    """Assert that using set_defaults overrides cors.* config values."""
    cors.set_defaults(**self.override_opts)
    self.application = cors.CORS(test_application, self.config)
    gc = self.config.cors
    self.assertEqual(['http://valid.example.com'], gc.allowed_origin)
    self.assertEqual(self.override_opts['allow_credentials'], gc.allow_credentials)
    self.assertEqual(self.override_opts['expose_headers'], gc.expose_headers)
    self.assertEqual(10, gc.max_age)
    self.assertEqual(self.override_opts['allow_methods'], gc.allow_methods)
    self.assertEqual(self.override_opts['allow_headers'], gc.allow_headers)
    cc = self.config['cors.override_creds']
    self.assertEqual(['http://creds.example.com'], cc.allowed_origin)
    self.assertTrue(cc.allow_credentials)
    self.assertEqual(self.override_opts['expose_headers'], cc.expose_headers)
    self.assertEqual(10, cc.max_age)
    self.assertEqual(self.override_opts['allow_methods'], cc.allow_methods)
    self.assertEqual(self.override_opts['allow_headers'], cc.allow_headers)
    ec = self.config['cors.override_headers']
    self.assertEqual(['http://headers.example.com'], ec.allowed_origin)
    self.assertEqual(self.override_opts['allow_credentials'], ec.allow_credentials)
    self.assertEqual(['X-Header-1', 'X-Header-2'], ec.expose_headers)
    self.assertEqual(10, ec.max_age)
    self.assertEqual(self.override_opts['allow_methods'], ec.allow_methods)
    self.assertEqual(['X-Header-1', 'X-Header-2'], ec.allow_headers)
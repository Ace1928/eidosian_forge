import webob
from glance.api.middleware import context
import glance.context
from glance.tests.unit import base
def test_invalid_service_catalog(self):
    catalog_json = 'bad json'
    req = self._build_request(service_catalog=catalog_json)
    middleware = self._build_middleware()
    self.assertRaises(webob.exc.HTTPInternalServerError, middleware.process_request, req)
import http.client as http
from oslo_serialization import jsonutils
import webob
from glance.common import auth
from glance.common import exception
from glance.tests import utils
def test_get_endpoint_raises_with_invalid_service_type(self):
    self.assertRaises(exception.NoServiceEndpoint, auth.get_endpoint, self.service_catalog, service_type='foo')
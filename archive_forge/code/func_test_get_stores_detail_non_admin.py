from oslo_config import cfg
import webob.exc
import glance.api.v2.discovery
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def test_get_stores_detail_non_admin(self):
    req = unit_test_utils.get_fake_request()
    self.assertRaises(webob.exc.HTTPForbidden, self.controller.get_stores_detail, req)
from oslo_config import cfg
import webob.exc
import glance.api.v2.discovery
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def test_get_stores(self):
    available_stores = ['cheap', 'fast', 'readonly_store', 'fast-cinder', 'fast-rbd', 'reliable']
    req = unit_test_utils.get_fake_request()
    output = self.controller.get_stores(req)
    self.assertIn('stores', output)
    for stores in output['stores']:
        self.assertIn('id', stores)
        self.assertNotIn('weight', stores)
        self.assertIn(stores['id'], available_stores)
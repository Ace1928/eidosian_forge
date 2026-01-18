from oslo_config import cfg
import webob.exc
import glance.api.v2.discovery
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def test_get_stores_detail(self):
    available_stores = ['cheap', 'fast', 'readonly_store', 'fast-cinder', 'fast-rbd', 'reliable']
    available_store_type = ['file', 'file', 'http', 'cinder', 'rbd', 'swift']
    req = unit_test_utils.get_fake_request(roles=['admin'])
    output = self.controller.get_stores_detail(req)
    self.assertEqual(len(CONF.enabled_backends), len(output['stores']))
    self.assertIn('stores', output)
    for stores in output['stores']:
        self.assertIn('id', stores)
        self.assertIn(stores['id'], available_stores)
        self.assertIn(stores['type'], available_store_type)
        self.assertIsNotNone(stores['properties'])
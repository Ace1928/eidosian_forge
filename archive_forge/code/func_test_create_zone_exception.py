import copy
from openstack import exceptions
from openstack.tests.unit import base
def test_create_zone_exception(self):
    self.register_uris([dict(method='POST', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones']), status_code=500)])
    self.assertRaises(exceptions.SDKException, self.cloud.create_zone, 'example.net.')
    self.assert_calls()
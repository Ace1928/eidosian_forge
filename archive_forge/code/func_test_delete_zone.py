import copy
from openstack import exceptions
from openstack.tests.unit import base
def test_delete_zone(self):
    fake_zone = ZoneTestWrapper(self, zone_dict)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones', fake_zone['id']]), json=fake_zone.get_get_response_json()), dict(method='DELETE', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones', fake_zone['id']]), status_code=202)])
    self.assertTrue(self.cloud.delete_zone(fake_zone['id']))
    self.assert_calls()
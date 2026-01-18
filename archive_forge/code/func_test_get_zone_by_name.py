import copy
from openstack import exceptions
from openstack.tests.unit import base
def test_get_zone_by_name(self):
    fake_zone = ZoneTestWrapper(self, zone_dict)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones', fake_zone['name']]), status_code=404), dict(method='GET', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones'], qs_elements=['name={name}'.format(name=fake_zone['name'])]), json={'zones': [fake_zone.get_get_response_json()]})])
    res = self.cloud.get_zone(fake_zone['name'])
    fake_zone.cmp(res)
    self.assert_calls()
from openstack import exceptions
from openstack.tests.unit import base
from openstack.tests.unit.cloud import test_zone
def test_get_recordset_not_found_returns_false(self):
    fake_zone = test_zone.ZoneTestWrapper(self, zone)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones', fake_zone['id']]), json=fake_zone.get_get_response_json()), dict(method='GET', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones', fake_zone['id'], 'recordsets', 'fake']), status_code=404), dict(method='GET', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones', fake_zone['id'], 'recordsets'], qs_elements=['name=fake']), json={'recordsets': []})])
    res = self.cloud.get_recordset(fake_zone['id'], 'fake')
    self.assertFalse(res)
    self.assert_calls()
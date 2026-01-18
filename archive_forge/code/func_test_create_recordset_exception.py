from openstack import exceptions
from openstack.tests.unit import base
from openstack.tests.unit.cloud import test_zone
def test_create_recordset_exception(self):
    fake_zone = test_zone.ZoneTestWrapper(self, zone)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones', fake_zone['id']]), json=fake_zone.get_get_response_json()), dict(method='POST', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones', zone['id'], 'recordsets']), status_code=500, validate=dict(json={'name': 'www2.example.net.', 'records': ['192.168.1.2'], 'type': 'A'}))])
    self.assertRaises(exceptions.SDKException, self.cloud.create_recordset, fake_zone['id'], 'www2.example.net.', 'a', ['192.168.1.2'])
    self.assert_calls()
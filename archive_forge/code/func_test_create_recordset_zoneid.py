from openstack import exceptions
from openstack.tests.unit import base
from openstack.tests.unit.cloud import test_zone
def test_create_recordset_zoneid(self):
    fake_zone = test_zone.ZoneTestWrapper(self, zone)
    fake_rs = RecordsetTestWrapper(self, recordset)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones', fake_zone['id']]), json=fake_zone.get_get_response_json()), dict(method='POST', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones', zone['id'], 'recordsets']), json=fake_rs.get_create_response_json(), validate=dict(json={'records': fake_rs['records'], 'type': fake_rs['type'], 'name': fake_rs['name'], 'description': fake_rs['description'], 'ttl': fake_rs['ttl']}))])
    rs = self.cloud.create_recordset(zone=fake_zone['id'], name=fake_rs['name'], recordset_type=fake_rs['type'], records=fake_rs['records'], description=fake_rs['description'], ttl=fake_rs['ttl'])
    fake_rs.cmp(rs)
    self.assert_calls()
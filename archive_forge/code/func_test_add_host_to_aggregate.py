from openstack.tests import fakes
from openstack.tests.unit import base
def test_add_host_to_aggregate(self):
    hostname = 'host1'
    self.register_uris([dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['os-aggregates', '1']), json=self.fake_aggregate), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['os-aggregates', '1', 'action']), json={'aggregate': self.fake_aggregate}, validate=dict(json={'add_host': {'host': hostname}}))])
    self.cloud.add_host_to_aggregate('1', hostname)
    self.assert_calls()
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_aggregate_with_az(self):
    availability_zone = 'az1'
    az_aggregate = fakes.make_fake_aggregate(1, self.aggregate_name, availability_zone=availability_zone)
    create_aggregate = az_aggregate.copy()
    del create_aggregate['metadata']
    del create_aggregate['hosts']
    self.register_uris([dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['os-aggregates']), json={'aggregate': create_aggregate}, validate=dict(json={'aggregate': {'name': self.aggregate_name, 'availability_zone': availability_zone}}))])
    self.cloud.create_aggregate(name=self.aggregate_name, availability_zone=availability_zone)
    self.assert_calls()
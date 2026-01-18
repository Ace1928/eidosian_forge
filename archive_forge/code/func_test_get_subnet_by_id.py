import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import subnet as _subnet
from openstack.tests.unit import base
def test_get_subnet_by_id(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'subnets', self.subnet_id]), json={'subnet': self.mock_subnet_rep})])
    r = self.cloud.get_subnet_by_id(self.subnet_id)
    self.assertIsNotNone(r)
    self._compare_subnets(self.mock_subnet_rep, r)
    self.assert_calls()
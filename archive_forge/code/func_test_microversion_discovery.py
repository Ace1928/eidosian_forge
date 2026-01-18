import ddt
from keystoneauth1 import exceptions
from openstack.tests.unit import base
def test_microversion_discovery(self):
    self.assertEqual((1, 17), self.cloud.placement.get_endpoint_data().max_microversion)
    self.assert_calls()
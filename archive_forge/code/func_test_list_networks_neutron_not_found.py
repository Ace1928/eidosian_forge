import copy
from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import network as _network
from openstack.tests.unit import base
def test_list_networks_neutron_not_found(self):
    self.use_nothing()
    self.cloud.has_service = mock.Mock(return_value=False)
    self.assertEqual([], self.cloud.list_networks())
    self.assert_calls()
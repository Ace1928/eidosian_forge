import copy
from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import network as _network
from openstack.tests.unit import base
def test_create_network_with_wrong_mtu_type(self):
    with testtools.ExpectedException(exceptions.SDKException, "Parameter 'mtu_size' must be an integer."):
        self.cloud.create_network('netname', mtu_size='fourty_two')
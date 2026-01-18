from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_no_reservation(self, mock_fetch):
    self.node.reservation = None
    node = self.node.wait_for_reservation(None)
    self.assertIs(node, self.node)
    self.assertFalse(mock_fetch.called)
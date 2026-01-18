from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_node_patch(self, mock_patch):
    patch = {'path': 'test'}
    self.node.patch(self.session, patch=patch)
    mock_patch.assert_called_once()
    kwargs = mock_patch.call_args[1]
    self.assertEqual(kwargs['patch'], {'path': 'test'})
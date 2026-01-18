from unittest import mock
from openstack.baremetal.v1 import _proxy
from openstack.baremetal.v1 import allocation
from openstack.baremetal.v1 import chassis
from openstack.baremetal.v1 import driver
from openstack.baremetal.v1 import node
from openstack.baremetal.v1 import port
from openstack.baremetal.v1 import port_group
from openstack.baremetal.v1 import volume_connector
from openstack.baremetal.v1 import volume_target
from openstack import exceptions
from openstack.tests.unit import base
from openstack.tests.unit import test_proxy_base
@mock.patch.object(node.Node, 'fetch', autospec=True)
def test__get_with_fields_node(self, mock_fetch):
    result = self.proxy._get_with_fields(node.Node, 'value', fields=['maintenance', 'id', 'instance_id'])
    self.assertIs(result, mock_fetch.return_value)
    mock_fetch.assert_called_once_with(mock.ANY, self.proxy, error_message=mock.ANY, fields='maintenance,uuid,instance_uuid')
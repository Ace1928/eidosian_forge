from unittest import mock
import swiftclient.client as sc
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine.resources.openstack.swift import container as swift_c
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
@mock.patch('swiftclient.client.Connection.put_container')
def test_delete_retain(self, mock_put):
    self.t['Resources']['SwiftContainer']['DeletionPolicy'] = 'Retain'
    stack = utils.parse_stack(self.t)
    container_name = utils.PhysName(stack.name, 'test_resource')
    container = self._create_container(stack)
    runner = scheduler.TaskRunner(container.delete)
    runner()
    self.assertEqual((container.DELETE, container.COMPLETE), container.state)
    mock_put.assert_called_once_with(container_name, {})
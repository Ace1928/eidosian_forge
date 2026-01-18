from unittest import mock
import swiftclient.client as sc
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine.resources.openstack.swift import container as swift_c
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
@mock.patch('swiftclient.client.Connection.delete_container')
@mock.patch('swiftclient.client.Connection.get_container')
@mock.patch('swiftclient.client.Connection.put_container')
def test_delete_exception(self, mock_put, mock_get, mock_delete):
    stack = utils.parse_stack(self.t)
    container_name = utils.PhysName(stack.name, 'test_resource')
    mock_delete.side_effect = sc.ClientException('test-delete-failure')
    mock_get.return_value = ({'name': container_name}, [])
    container = self._create_container(stack)
    runner = scheduler.TaskRunner(container.delete)
    self.assertRaises(exception.ResourceFailure, runner)
    self.assertEqual((container.DELETE, container.FAILED), container.state)
    mock_put.assert_called_once_with(container_name, {})
    mock_get.assert_called_once_with(container_name)
    mock_delete.assert_called_once_with(container_name)
from unittest import mock
import swiftclient.client as sc
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine.resources.openstack.swift import container as swift_c
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
@mock.patch('swiftclient.client.Connection.get_container')
@mock.patch('swiftclient.client.Connection.put_container')
def test_delete_non_empty_not_allowed(self, mock_put, mock_get):
    stack = utils.parse_stack(self.t)
    container_name = utils.PhysName(stack.name, 'test_resource')
    mock_get.return_value = ({'name': container_name}, [{'name': 'test_object'}])
    container = self._create_container(stack)
    runner = scheduler.TaskRunner(container.delete)
    ex = self.assertRaises(exception.ResourceFailure, runner)
    self.assertEqual((container.DELETE, container.FAILED), container.state)
    self.assertIn('ResourceActionNotSupported: resources.test_resource: Deleting non-empty container', str(ex))
    mock_put.assert_called_once_with(container_name, {})
    mock_get.assert_called_once_with(container_name)
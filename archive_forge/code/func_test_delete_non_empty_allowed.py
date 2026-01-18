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
@mock.patch('swiftclient.client.Connection.delete_object')
@mock.patch('swiftclient.client.Connection.get_container')
@mock.patch('swiftclient.client.Connection.put_container')
def test_delete_non_empty_allowed(self, mock_put, mock_get, mock_delete_object, mock_delete_container):
    res_prop = self.t['Resources']['SwiftContainer']['Properties']
    res_prop['PurgeOnDelete'] = True
    stack = utils.parse_stack(self.t)
    container_name = utils.PhysName(stack.name, 'test_resource')
    get_return_values = [({'name': container_name}, [{'name': 'test_object1'}, {'name': 'test_object2'}]), ({'name': container_name}, [{'name': 'test_object1'}])]
    mock_get.side_effect = get_return_values
    container = self._create_container(stack)
    runner = scheduler.TaskRunner(container.delete)
    runner()
    self.assertEqual((container.DELETE, container.COMPLETE), container.state)
    mock_put.assert_called_once_with(container_name, {})
    mock_delete_container.assert_called_once_with(container_name)
    self.assertEqual(2, mock_get.call_count)
    self.assertEqual(2, mock_delete_object.call_count)
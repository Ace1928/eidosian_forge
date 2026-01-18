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
def test_delete_non_empty_allowed_not_found(self, mock_put, mock_get, mock_delete_object, mock_delete_container):
    res_prop = self.t['Resources']['SwiftContainer']['Properties']
    res_prop['PurgeOnDelete'] = True
    stack = utils.parse_stack(self.t)
    container_name = utils.PhysName(stack.name, 'test_resource')
    mock_get.return_value = ({'name': container_name}, [{'name': 'test_object'}])
    mock_delete_object.side_effect = sc.ClientException('object-is-gone', http_status=404)
    mock_delete_container.side_effect = sc.ClientException('container-is-gone', http_status=404)
    container = self._create_container(stack)
    runner = scheduler.TaskRunner(container.delete)
    runner()
    self.assertEqual((container.DELETE, container.COMPLETE), container.state)
    mock_put.assert_called_once_with(container_name, {})
    mock_get.assert_called_once_with(container_name)
    mock_delete_object.assert_called_once_with(container_name, 'test_object')
    mock_delete_container.assert_called_once_with(container_name)
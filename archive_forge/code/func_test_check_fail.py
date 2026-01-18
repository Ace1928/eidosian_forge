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
def test_check_fail(self, mock_put, mock_get):
    res_prop = self.t['Resources']['SwiftContainer']['Properties']
    res_prop['PurgeOnDelete'] = True
    stack = utils.parse_stack(self.t)
    mock_get.side_effect = Exception('boom')
    container = self._create_container(stack)
    runner = scheduler.TaskRunner(container.check)
    ex = self.assertRaises(exception.ResourceFailure, runner)
    self.assertIn('boom', str(ex))
    self.assertEqual((container.CHECK, container.FAILED), container.state)
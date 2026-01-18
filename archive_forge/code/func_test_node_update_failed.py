import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import node as sn
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
from openstack import exceptions
def test_node_update_failed(self):
    node = self._create_node()
    new_t = copy.deepcopy(self.t)
    props = new_t['resources']['senlin-node']['properties']
    props['name'] = 'new_name'
    rsrc_defns = template.Template(new_t).resource_definitions(self.stack)
    new_node = rsrc_defns['senlin-node']
    self.senlin_mock.update_node.return_value = mock.Mock(location='/actions/fake-action')
    self.senlin_mock.get_action.return_value = mock.Mock(status='FAILED', status_reason='oops')
    update_task = scheduler.TaskRunner(node.update, new_node)
    ex = self.assertRaises(exception.ResourceFailure, update_task)
    expected = 'ResourceInError: resources.senlin-node: Went to status FAILED due to "oops"'
    self.assertEqual(expected, str(ex))
    self.assertEqual((node.UPDATE, node.FAILED), node.state)
    self.assertEqual(2, self.senlin_mock.get_action.call_count)
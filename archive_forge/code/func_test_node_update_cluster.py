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
def test_node_update_cluster(self):
    node = self._create_node()
    self.senlin_mock.get_cluster.side_effect = [mock.Mock(id='new_cluster_id'), mock.Mock(id='fake_cluster_id'), mock.Mock(id='new_cluster_id')]
    new_t = copy.deepcopy(self.t)
    props = new_t['resources']['senlin-node']['properties']
    props['cluster'] = 'new_cluster'
    rsrc_defns = template.Template(new_t).resource_definitions(self.stack)
    new_node = rsrc_defns['senlin-node']
    self.senlin_mock.remove_nodes_from_cluster.return_value = {'action': 'remove_node_from_cluster'}
    self.senlin_mock.add_nodes_to_cluster.return_value = {'action': 'add_node_to_cluster'}
    scheduler.TaskRunner(node.update, new_node)()
    self.assertEqual((node.UPDATE, node.COMPLETE), node.state)
    self.senlin_mock.remove_nodes_from_cluster.assert_called_once_with(cluster='fake_cluster_id', nodes=[node.resource_id])
    self.senlin_mock.add_nodes_to_cluster.assert_called_once_with(cluster='new_cluster_id', nodes=[node.resource_id])
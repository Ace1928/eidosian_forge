import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import cluster as sc
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
from openstack import exceptions
def test_cluster_update_desire_capacity(self):
    cluster = self._create_cluster(self.t)
    new_t = copy.deepcopy(self.t)
    props = new_t['resources']['senlin-cluster']['properties']
    props['desired_capacity'] = 10
    rsrc_defns = template.Template(new_t).resource_definitions(self.stack)
    new_cluster = rsrc_defns['senlin-cluster']
    self.senlin_mock.resize_cluster.return_value = {'action': 'fake-action'}
    self.senlin_mock.get_action.return_value = mock.Mock(status='SUCCEEDED')
    scheduler.TaskRunner(cluster.update, new_cluster)()
    self.assertEqual((cluster.UPDATE, cluster.COMPLETE), cluster.state)
    cluster_resize_kwargs = {'adjustment_type': 'EXACT_CAPACITY', 'number': 10}
    self.senlin_mock.resize_cluster.assert_called_once_with(cluster=cluster.resource_id, **cluster_resize_kwargs)
    self.assertEqual(2, self.senlin_mock.get_action.call_count)
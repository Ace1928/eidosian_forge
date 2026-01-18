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
def test_cluster_update_failed(self):
    cluster = self._create_cluster(self.t)
    new_t = copy.deepcopy(self.t)
    props = new_t['resources']['senlin-cluster']['properties']
    props['desired_capacity'] = 3
    rsrc_defns = template.Template(new_t).resource_definitions(self.stack)
    update_snippet = rsrc_defns['senlin-cluster']
    self.senlin_mock.resize_cluster.return_value = {'action': 'fake-action'}
    self.senlin_mock.get_action.return_value = mock.Mock(status='FAILED', status_reason='Unknown')
    exc = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(cluster.update, update_snippet))
    self.assertEqual('ResourceInError: resources.senlin-cluster: Went to status FAILED due to "Unknown"', str(exc))
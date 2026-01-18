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
def test_cluster_delete_error(self):
    cluster = self._create_cluster(self.t)
    self.senlin_mock.get_cluster.side_effect = exception.Error('oops')
    delete_task = scheduler.TaskRunner(cluster.delete)
    ex = self.assertRaises(exception.ResourceFailure, delete_task)
    expected = 'Error: resources.senlin-cluster: oops'
    self.assertEqual(expected, str(ex))
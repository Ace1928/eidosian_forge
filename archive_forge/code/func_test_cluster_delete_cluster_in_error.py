from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import cluster as sc
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_cluster_delete_cluster_in_error(self):
    cluster = self._create_cluster(self.t)
    self.cl_mgr.get.side_effect = [self.fake_cl, FakeCluster(status='Error')]
    self.cl_mgr.get.reset_mock()
    delete_task = scheduler.TaskRunner(cluster.delete)
    ex = self.assertRaises(exception.ResourceFailure, delete_task)
    expected = 'ResourceInError: resources.super-cluster: Went to status Error due to "Unknown"'
    self.assertEqual(expected, str(ex))
    self.cl_mgr.delete.assert_called_once_with(self.fake_cl.id)
    self.assertEqual(2, self.cl_mgr.get.call_count)
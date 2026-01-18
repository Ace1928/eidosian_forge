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
def test_cluster_create_fails(self):
    cfg.CONF.set_override('action_retry_limit', 0)
    cluster = self._init_cluster(self.t)
    self.cl_mgr.create.return_value = self.fake_cl
    self.cl_mgr.get.return_value = FakeCluster(status='Error')
    create_task = scheduler.TaskRunner(cluster.create)
    ex = self.assertRaises(exception.ResourceFailure, create_task)
    expected = 'ResourceInError: resources.super-cluster: Went to status Error due to "Unknown"'
    self.assertEqual(expected, str(ex))
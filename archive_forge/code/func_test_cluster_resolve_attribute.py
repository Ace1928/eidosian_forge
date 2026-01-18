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
def test_cluster_resolve_attribute(self):
    cluster = self._create_cluster(self.t)
    self.cl_mgr.get.reset_mock()
    self.assertEqual(self.fake_cl.info, cluster._resolve_attribute('info'))
    self.assertEqual(self.fake_cl.status, cluster._resolve_attribute('status'))
    self.assertEqual({'cluster': 'info'}, cluster.FnGetAtt('show'))
    self.assertEqual(3, self.cl_mgr.get.call_count)
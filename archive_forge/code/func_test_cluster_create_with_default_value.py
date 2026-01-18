import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import magnum as mc
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine.resources.openstack.magnum import cluster
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_cluster_create_with_default_value(self):
    b = self._create_resource('cluster', self.min_rsrc_defn, self.stack)
    self.assertEqual(None, b.properties.get(cluster.Cluster.NAME))
    self.assertEqual(self.fake_cluster_template, b.properties.get(cluster.Cluster.CLUSTER_TEMPLATE))
    self.assertEqual(None, b.properties.get(cluster.Cluster.KEYPAIR))
    self.assertEqual(1, b.properties.get(cluster.Cluster.NODE_COUNT))
    self.assertEqual(1, b.properties.get(cluster.Cluster.MASTER_COUNT))
    self.assertEqual(None, b.properties.get(cluster.Cluster.DISCOVERY_URL))
    self.assertEqual(60, b.properties.get(cluster.Cluster.CREATE_TIMEOUT))
    scheduler.TaskRunner(b.create)()
    self.assertEqual(self.resource_id, b.resource_id)
    self.assertEqual((b.CREATE, b.COMPLETE), b.state)
    self.client.clusters.create.assert_called_once_with(name=None, keypair=None, cluster_template_id=self.fake_cluster_template, node_count=1, master_count=1, discovery_url=None, create_timeout=60)
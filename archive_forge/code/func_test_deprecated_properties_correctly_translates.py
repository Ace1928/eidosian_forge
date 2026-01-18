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
def test_deprecated_properties_correctly_translates(self):
    tmpl = '\nheat_template_version: 2013-05-23\ndescription: Hadoop Cluster by Sahara\nresources:\n  super-cluster:\n    type: OS::Sahara::Cluster\n    properties:\n      name: super-cluster\n      plugin_name: vanilla\n      hadoop_version: 2.3.0\n      cluster_template_id: some_cluster_template_id\n      image: some_image\n      key_name: admin\n      neutron_management_network: some_network\n      '
    ct = self._create_cluster(template_format.parse(tmpl))
    self.assertEqual('some_image_id', ct.properties.get('default_image_id'))
    self.assertIsNone(ct.properties.get('image_id'))
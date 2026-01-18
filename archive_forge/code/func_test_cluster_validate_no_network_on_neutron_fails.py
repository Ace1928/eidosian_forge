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
def test_cluster_validate_no_network_on_neutron_fails(self):
    self.t['resources']['super-cluster']['properties'].pop('neutron_management_network')
    cluster = self._init_cluster(self.t)
    ex = self.assertRaises(exception.StackValidationFailed, cluster.validate)
    error_msg = 'Property error: resources.super-cluster.properties: Property neutron_management_network not assigned'
    self.assertEqual(error_msg, str(ex))
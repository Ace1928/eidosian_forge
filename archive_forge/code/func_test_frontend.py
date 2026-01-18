import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.aws.lb import loadbalancer as lb
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_frontend(self):
    props = {'HealthCheck': {}, 'Listeners': [{'LoadBalancerPort': 4014}]}
    self._mock_props(props)
    exp = '\nfrontend http\n    bind *:4014\n    default_backend servers\n'
    actual = self.lb._haproxy_config_frontend()
    self.assertEqual(exp, actual)
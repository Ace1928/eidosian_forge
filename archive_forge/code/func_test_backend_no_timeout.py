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
def test_backend_no_timeout(self):
    self._mock_props({'HealthCheck': None})
    be = self.lb._haproxy_config_backend()
    exp = '\nbackend servers\n    balance roundrobin\n    option http-server-close\n    option forwardfor\n    option httpchk\n\n'
    self.assertEqual(exp, be)
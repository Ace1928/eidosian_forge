import copy
from oslo_log import log as logging
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import trunk
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests import common
from heat.tests import utils
from neutronclient.common import exceptions as ncex
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
def test_create_degraded(self):
    t = template_format.parse(create_template)
    stack = utils.parse_stack(t)
    rv = {'trunk': {'id': 'trunk id', 'status': 'DEGRADED'}}
    self.create_trunk_mock.return_value = rv
    self.show_trunk_mock.return_value = rv
    trunk = stack['trunk']
    e = self.assertRaises(exception.ResourceInError, trunk.check_create_complete, trunk.resource_id)
    self.assertIn('Went to status DEGRADED due to', str(e))
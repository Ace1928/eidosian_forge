import copy
from unittest import mock
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception as heat_ex
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.nova import floatingip
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def prepare_floating_ip(self):
    self.mock_create_floatingip()
    template = template_format.parse(floating_ip_template)
    self.stack = utils.parse_stack(template)
    defns = self.stack.t.resource_definitions(self.stack)
    return floatingip.NovaFloatingIp('MyFloatingIP', defns['MyFloatingIP'], self.stack)
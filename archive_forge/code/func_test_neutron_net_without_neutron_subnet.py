from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.manila import share_network
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_neutron_net_without_neutron_subnet(self):
    t = template_format.parse(stack_template)
    del t['resources']['share_network']['properties']['neutron_subnet']
    stack = utils.parse_stack(t)
    rsrc_defn = stack.t.resource_definitions(stack)['share_network']
    net = self._create_network('share_network', rsrc_defn, stack)
    msg = 'neutron_network cannot be specified without neutron_subnet.'
    self.assertRaisesRegex(exception.ResourcePropertyDependency, msg, net.validate)
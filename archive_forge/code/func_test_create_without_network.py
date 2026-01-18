from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.manila import share_network
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_create_without_network(self):
    t = template_format.parse(stack_template)
    del t['resources']['share_network']['properties']['neutron_network']
    stack = utils.parse_stack(t)
    rsrc_defn = stack.t.resource_definitions(stack)['share_network']
    net = self._create_network('share_network', rsrc_defn, stack)
    self.assertEqual((net.CREATE, net.COMPLETE), net.state)
    self.assertEqual('42', net.resource_id)
    net.client().share_networks.create.assert_called_with(name='1', description='2', neutron_net_id='3', neutron_subnet_id='4', nova_net_id=None)
    calls = [mock.call('42', '6'), mock.call('42', '7')]
    net.client().share_networks.add_security_service.assert_has_calls(calls, any_order=True)
    self.assertEqual('share_networks', net.entity)
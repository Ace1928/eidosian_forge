import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import openstacksdk
from heat.engine.hot import functions as hot_funcs
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.neutron import subnet
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests import common
from heat.tests import utils
def test_ipv6_subnet(self):
    t = template_format.parse(neutron_template)
    props = t['resources']['sub_net']['properties']
    props.pop('allocation_pools')
    props.pop('host_routes')
    props['ip_version'] = 6
    props['ipv6_address_mode'] = 'slaac'
    props['ipv6_ra_mode'] = 'slaac'
    props['cidr'] = 'fdfa:6a50:d22b::/64'
    props['dns_nameservers'] = ['2001:4860:4860::8844']
    stack = utils.parse_stack(t)
    create_info = {'subnet': {'name': utils.PhysName(stack.name, 'test_subnet'), 'network_id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'dns_nameservers': [u'2001:4860:4860::8844'], 'ip_version': 6, 'enable_dhcp': True, 'cidr': u'fdfa:6a50:d22b::/64', 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'ipv6_address_mode': 'slaac', 'ipv6_ra_mode': 'slaac'}}
    subnet_info = copy.deepcopy(create_info)
    subnet_info['subnet']['id'] = '91e47a57-7508-46fe-afc9-fc454e8580e1'
    self.create_mock.return_value = subnet_info
    self.patchobject(stack['net'], 'FnGetRefId', return_value='fc68ea2c-b60b-4b4f-bd82-94ec81110766')
    rsrc = self.create_subnet(t, stack, 'sub_net')
    scheduler.TaskRunner(rsrc.create)()
    self.create_mock.assert_called_once_with(create_info)
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    rsrc.validate()
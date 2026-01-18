import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import router
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_create_router_gateway_enable_snat(self):
    self.find_rsrc_mock.side_effect = ['fc68ea2c-b60b-4b4f-bd82-94ec81110766']
    router_info = {'router': {'name': 'Test Router', 'external_gateway_info': {'network_id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}, 'admin_state_up': True, 'status': 'BUILD', 'id': '3e46229d-8fce-4733-819a-b5fe630550f8'}}
    active_info = copy.deepcopy(router_info)
    active_info['router']['status'] = 'ACTIVE'
    self.create_mock.return_value = router_info
    self.show_mock.side_effect = [active_info, active_info]
    t = template_format.parse(neutron_external_gateway_template)
    t['resources']['router']['properties']['external_gateway_info'].pop('enable_snat')
    t['resources']['router']['properties']['external_gateway_info'].pop('external_fixed_ips')
    stack = utils.parse_stack(t)
    rsrc = self.create_router(t, stack, 'router')
    self.create_mock.assert_called_with({'router': {'name': 'Test Router', 'external_gateway_info': {'network_id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}, 'admin_state_up': True}})
    rsrc.validate()
    ref_id = rsrc.FnGetRefId()
    self.assertEqual('3e46229d-8fce-4733-819a-b5fe630550f8', ref_id)
    gateway_info = rsrc.FnGetAtt('external_gateway_info')
    self.assertEqual('fc68ea2c-b60b-4b4f-bd82-94ec81110766', gateway_info.get('network_id'))
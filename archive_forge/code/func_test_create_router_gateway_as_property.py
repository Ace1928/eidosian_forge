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
def test_create_router_gateway_as_property(self):
    t, stack = self._test_router_with_gateway()
    rsrc = self.create_router(t, stack, 'router')
    self._assert_mock_call_create_with_router_gw()
    ref_id = rsrc.FnGetRefId()
    self.assertEqual('3e46229d-8fce-4733-819a-b5fe630550f8', ref_id)
    gateway_info = rsrc.FnGetAtt('external_gateway_info')
    self.assertEqual('fc68ea2c-b60b-4b4f-bd82-94ec81110766', gateway_info.get('network_id'))
    self.assertTrue(gateway_info.get('enable_snat'))
    self.assertEqual([{'subnet_id': 'sub1234', 'ip_address': '192.168.10.99'}], gateway_info.get('external_fixed_ips'))
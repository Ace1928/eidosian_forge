import copy
import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallgroup
from neutronclient.osc.v2 import utils as v2_utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
def test_create_with_all_params(self):
    name = 'my-name'
    description = 'my-desc'
    ingress_policy = 'my-ingress-policy'
    egress_policy = 'my-egress-policy'

    def _mock_find(*args, **kwargs):
        return {'id': args[0]}
    self.networkclient.find_firewall_policy.side_effect = _mock_find
    port = 'port'
    self.networkclient.find_port.side_effect = _mock_find
    tenant_id = 'my-tenant'
    arglist = ['--name', name, '--description', description, '--ingress-firewall-policy', ingress_policy, '--egress-firewall-policy', egress_policy, '--port', port, '--project', tenant_id, '--share', '--disable']
    verifylist = [('name', name), ('description', description), ('ingress_firewall_policy', ingress_policy), ('egress_firewall_policy', egress_policy), ('port', [port]), ('share', True), ('project', tenant_id), ('disable', True)]
    request, response = _generate_req_and_res(verifylist)
    self._update_expect_response(request, response)
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    headers, data = self.cmd.take_action(parsed_args)
    self.check_results(headers, data, request)
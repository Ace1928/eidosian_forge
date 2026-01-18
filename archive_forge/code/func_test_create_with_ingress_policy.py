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
def test_create_with_ingress_policy(self):
    ingress_policy = 'my-ingress-policy'

    def _mock_port_fwg(*args, **kwargs):
        return {'id': args[0]}
    self.networkclient.find_firewall_policy.side_effect = _mock_port_fwg
    arglist = ['--ingress-firewall-policy', ingress_policy]
    verifylist = [('ingress_firewall_policy', ingress_policy)]
    request, response = _generate_req_and_res(verifylist)
    self._update_expect_response(request, response)
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    headers, data = self.cmd.take_action(parsed_args)
    self.networkclient.find_firewall_policy.assert_called_once_with(ingress_policy)
    self.check_results(headers, data, request)
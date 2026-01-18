import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallpolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
def test_create_with_rules(self):
    name = 'my-fwg'
    rule1 = 'rule1'
    rule2 = 'rule2'

    def _mock_policy(*args, **kwargs):
        return {'id': args[0]}
    self.networkclient.find_firewall_rule.side_effect = _mock_policy
    arglist = [name, '--firewall-rule', rule1, '--firewall-rule', rule2]
    verifylist = [('name', name), ('firewall_rule', [rule1, rule2])]
    request, response = _generate_req_and_res(verifylist)
    self._update_expect_response(request, response)
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    headers, data = self.cmd.take_action(parsed_args)
    self.assertEqual(2, self.networkclient.find_firewall_rule.call_count)
    self.check_results(headers, data, request)
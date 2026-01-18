import testtools
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
def test_delete_with_one_resource(self):
    target = self.resource['id']

    def _mock_fwaas(*args, **kwargs):
        return {'id': args[0]}
    self.networkclient.find_firewall_group.side_effect = _mock_fwaas
    self.networkclient.find_firewall_policy.side_effect = _mock_fwaas
    self.networkclient.find_firewall_rule.side_effect = _mock_fwaas
    arglist = [target]
    verifylist = [(self.res, [target])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.mocked.assert_called_once_with(target)
    self.assertIsNone(result)
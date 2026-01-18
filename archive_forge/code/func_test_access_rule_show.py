from unittest import mock
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_access_rules as osc_share_access_rules
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_access_rule_show(self):
    arglist = [self.access_rule.id]
    verifylist = [('access_id', self.access_rule.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.access_rules_mock.get.assert_called_with(self.access_rule.id)
    self.assertEqual(ACCESS_RULE_ATTRIBUTES, columns)
    self.assertEqual(tuple(self.access_rule._info.values()), data)
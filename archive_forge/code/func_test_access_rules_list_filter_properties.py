from unittest import mock
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_access_rules as osc_share_access_rules
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_access_rules_list_filter_properties(self):
    arglist = [self.share.id, '--properties', 'key=value']
    verifylist = [('share', self.share.id), ('properties', ['key=value'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.shares_mock.get.assert_called_with(self.share.id)
    self.access_rules_mock.access_list.assert_called_with(self.share, {'metadata': {'key': 'value'}})
    self.assertEqual(self.access_rules_columns, columns)
    self.assertEqual(tuple(self.values_list), tuple(data))
from unittest import mock
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_access_rules as osc_share_access_rules
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.data({'access_to': '10.0.0.0/0', 'access_type': 'ip'}, {'access_key': '10.0.0.0/0', 'access_level': 'rw'})
def test_access_rules_list_access_filters(self, filters):
    arglist = [self.share.id]
    verifylist = [('share', self.share.id)]
    for filter_key, filter_value in filters.items():
        filter_arg = filter_key.replace('_', '-')
        arglist.append(f'--{filter_arg}')
        arglist.append(filter_value)
        verifylist.append((filter_key, filter_value))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.shares_mock.get.assert_called_with(self.share.id)
    self.access_rules_mock.access_list.assert_called_with(self.share, filters)
    self.assertEqual(self.access_rules_columns, columns)
    self.assertEqual(tuple(self.values_list), tuple(data))
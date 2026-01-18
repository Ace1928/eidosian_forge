from unittest import mock
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_access_rules as osc_share_access_rules
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.data({'access_to': '10.0.0.0/0', 'access_type': 'ip'}, {'access_key': '10.0.0.0/0', 'access_level': 'rw'})
def test_access_rules_list_access_filters_command_error(self, filters):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.81')
    arglist = [self.share.id]
    verifylist = [('share', self.share.id)]
    for filter_key, filter_value in filters.items():
        filter_arg = filter_key.replace('_', '-')
        arglist.append(f'--{filter_arg}')
        arglist.append(filter_value)
        verifylist.append((filter_key, filter_value))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
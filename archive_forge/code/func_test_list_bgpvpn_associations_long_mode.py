import operator
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def test_list_bgpvpn_associations_long_mode(self):
    count = 3
    fake_bgpvpn = fakes.create_one_bgpvpn()
    fake_res = fakes.create_resources(count=count)
    fake_res_assocs = fakes.create_resource_associations(fake_res)
    self.networkclient.bgpvpn_router_associations = mock.Mock(return_value=fake_res_assocs)
    arglist = ['--long', fake_bgpvpn['id']]
    verifylist = [('long', True), ('bgpvpn', fake_bgpvpn['id'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    headers, data = self.cmd.take_action(parsed_args)
    self.networkclient.bgpvpn_router_associations.assert_called_once_with(fake_bgpvpn['id'], retrieve_all=True)
    self.assertEqual(headers, list(headers_long))
    self.assertEqual(list(data), [_get_data(fake_res_assoc, columns_long) for fake_res_assoc in fake_res_assocs])
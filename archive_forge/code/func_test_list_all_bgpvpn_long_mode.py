import copy
import operator
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.osc.v2.networking_bgpvpn import bgpvpn
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def test_list_all_bgpvpn_long_mode(self):
    count = 3
    fake_bgpvpns = fakes.create_bgpvpns(count=count)
    self.networkclient.bgpvpns = mock.Mock(return_value=fake_bgpvpns)
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    headers, data = self.cmd.take_action(parsed_args)
    self.networkclient.bgpvpns.assert_called_once()
    self.assertEqual(headers, list(headers_long))
    self.assertListItemEqual(list(data), [_get_data(fake_bgpvpn, columns_long) for fake_bgpvpn in fake_bgpvpns])
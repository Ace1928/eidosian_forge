import copy
import operator
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.osc.v2.networking_bgpvpn import bgpvpn
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def test_create_bgpvpn_with_no_args(self):
    fake_bgpvpn = fakes.create_one_bgpvpn()
    self.networkclient.create_bgpvpn = mock.Mock(return_value=fake_bgpvpn)
    arglist = []
    verifylist = [('project', None), ('name', None), ('type', 'l3'), ('vni', None), ('local_pref', None), ('route_targets', None), ('import_targets', None), ('export_targets', None), ('route_distinguishers', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    cols, data = self.cmd.take_action(parsed_args)
    self.networkclient.create_bgpvpn.assert_called_once_with(**{'type': 'l3'})
    self.assertEqual(sorted(sorted_columns), sorted(cols))
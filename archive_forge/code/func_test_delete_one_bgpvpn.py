import copy
import operator
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.osc.v2.networking_bgpvpn import bgpvpn
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def test_delete_one_bgpvpn(self):
    fake_bgpvpn = fakes.create_one_bgpvpn()
    self.networkclient.delete_bgpvpn = mock.Mock()
    arglist = [fake_bgpvpn['id']]
    verifylist = [('bgpvpns', [fake_bgpvpn['id']])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.networkclient.delete_bgpvpn.assert_called_once_with(fake_bgpvpn['id'])
    self.assertIsNone(result)
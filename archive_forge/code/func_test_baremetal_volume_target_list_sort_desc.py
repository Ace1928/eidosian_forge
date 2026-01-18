import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_target as bm_vol_target
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_target_list_sort_desc(self):
    arglist = ['--sort', 'boot_index:desc']
    verifylist = [('sort', 'boot_index:desc')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    kwargs = {'marker': None, 'limit': None}
    self.baremetal_mock.volume_target.list.assert_called_once_with(**kwargs)
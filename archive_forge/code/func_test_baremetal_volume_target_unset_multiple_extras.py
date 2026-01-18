import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_target as bm_vol_target
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_target_unset_multiple_extras(self):
    arglist = [baremetal_fakes.baremetal_volume_target_uuid, '--extra', 'key1', '--extra', 'key2']
    verifylist = [('volume_target', baremetal_fakes.baremetal_volume_target_uuid), ('extra', ['key1', 'key2'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.volume_target.update.assert_called_once_with(baremetal_fakes.baremetal_volume_target_uuid, [{'path': '/extra/key1', 'op': 'remove'}, {'path': '/extra/key2', 'op': 'remove'}])
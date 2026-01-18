import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_target as bm_vol_target
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_target_set_multiple_properties(self):
    arglist = [baremetal_fakes.baremetal_volume_target_uuid, '--property', 'key1=val1', '--property', 'key2=val2']
    verifylist = [('volume_target', baremetal_fakes.baremetal_volume_target_uuid), ('properties', ['key1=val1', 'key2=val2'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.volume_target.update.assert_called_once_with(baremetal_fakes.baremetal_volume_target_uuid, [{'path': '/properties/key1', 'value': 'val1', 'op': 'add'}, {'path': '/properties/key2', 'value': 'val2', 'op': 'add'}])
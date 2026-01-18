import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_target as bm_vol_target
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_target_show(self):
    arglist = ['vvv-tttttt-vvvv']
    verifylist = [('volume_target', baremetal_fakes.baremetal_volume_target_uuid)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    args = ['vvv-tttttt-vvvv']
    self.baremetal_mock.volume_target.get.assert_called_once_with(*args, fields=None)
    collist = ('boot_index', 'extra', 'node_uuid', 'properties', 'uuid', 'volume_id', 'volume_type')
    self.assertEqual(collist, columns)
    datalist = (baremetal_fakes.baremetal_volume_target_boot_index, baremetal_fakes.baremetal_volume_target_extra, baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_volume_target_properties, baremetal_fakes.baremetal_volume_target_uuid, baremetal_fakes.baremetal_volume_target_volume_id, baremetal_fakes.baremetal_volume_target_volume_type)
    self.assertEqual(datalist, tuple(data))
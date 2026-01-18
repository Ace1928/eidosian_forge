import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_target as bm_vol_target
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_target_show_fields_multiple(self):
    arglist = ['vvv-tttttt-vvvv', '--fields', 'uuid', 'volume_id', '--fields', 'volume_type']
    verifylist = [('fields', [['uuid', 'volume_id'], ['volume_type']]), ('volume_target', baremetal_fakes.baremetal_volume_target_uuid)]
    fake_vt = copy.deepcopy(baremetal_fakes.VOLUME_TARGET)
    fake_vt.pop('node_uuid')
    fake_vt.pop('boot_index')
    fake_vt.pop('extra')
    fake_vt.pop('properties')
    self.baremetal_mock.volume_target.get.return_value = baremetal_fakes.FakeBaremetalResource(None, fake_vt, loaded=True)
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    args = ['vvv-tttttt-vvvv']
    fields = ['uuid', 'volume_id', 'volume_type']
    self.baremetal_mock.volume_target.get.assert_called_once_with(*args, fields=fields)
    collist = ('uuid', 'volume_id', 'volume_type')
    self.assertEqual(collist, columns)
    datalist = (baremetal_fakes.baremetal_volume_target_uuid, baremetal_fakes.baremetal_volume_target_volume_id, baremetal_fakes.baremetal_volume_target_volume_type)
    self.assertEqual(datalist, tuple(data))
import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_target as bm_vol_target
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_target_list_fields_multiple(self):
    arglist = ['--fields', 'uuid', 'boot_index', '--fields', 'extra']
    verifylist = [('fields', [['uuid', 'boot_index'], ['extra']])]
    fake_vt = copy.deepcopy(baremetal_fakes.VOLUME_TARGET)
    fake_vt.pop('volume_type')
    fake_vt.pop('properties')
    fake_vt.pop('volume_id')
    fake_vt.pop('node_uuid')
    self.baremetal_mock.volume_target.list.return_value = [baremetal_fakes.FakeBaremetalResource(None, fake_vt, loaded=True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'detail': False, 'marker': None, 'limit': None, 'fields': ('uuid', 'boot_index', 'extra')}
    self.baremetal_mock.volume_target.list.assert_called_with(**kwargs)
    collist = ('UUID', 'Boot Index', 'Extra')
    self.assertEqual(collist, columns)
    datalist = ((baremetal_fakes.baremetal_volume_target_uuid, baremetal_fakes.baremetal_volume_target_boot_index, baremetal_fakes.baremetal_volume_target_extra),)
    self.assertEqual(datalist, tuple(data))
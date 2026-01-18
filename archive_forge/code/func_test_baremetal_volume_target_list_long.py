import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_target as bm_vol_target
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_target_list_long(self):
    arglist = ['--long']
    verifylist = [('detail', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'detail': True, 'marker': None, 'limit': None}
    self.baremetal_mock.volume_target.list.assert_called_with(**kwargs)
    collist = ('UUID', 'Node UUID', 'Driver Volume Type', 'Properties', 'Boot Index', 'Extra', 'Volume ID', 'Created At', 'Updated At')
    self.assertEqual(collist, columns)
    datalist = ((baremetal_fakes.baremetal_volume_target_uuid, baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_volume_target_volume_type, baremetal_fakes.baremetal_volume_target_properties, baremetal_fakes.baremetal_volume_target_boot_index, baremetal_fakes.baremetal_volume_target_extra, baremetal_fakes.baremetal_volume_target_volume_id, '', ''),)
    self.assertEqual(datalist, tuple(data))
import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_list_long(self):
    arglist = ['--long']
    verifylist = [('detail', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'detail': True, 'marker': None, 'limit': None}
    self.baremetal_mock.portgroup.list.assert_called_with(**kwargs)
    collist = ('UUID', 'Address', 'Created At', 'Extra', 'Standalone Ports Supported', 'Node UUID', 'Name', 'Updated At', 'Internal Info', 'Mode', 'Properties')
    self.assertEqual(collist, columns)
    datalist = ((baremetal_fakes.baremetal_portgroup_uuid, baremetal_fakes.baremetal_portgroup_address, '', baremetal_fakes.baremetal_portgroup_extra, '', baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_portgroup_name, '', '', baremetal_fakes.baremetal_portgroup_mode, baremetal_fakes.baremetal_portgroup_properties),)
    self.assertEqual(datalist, tuple(data))
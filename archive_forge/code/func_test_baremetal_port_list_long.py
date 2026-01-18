import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_port_list_long(self):
    arglist = ['--long']
    verifylist = [('detail', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'detail': True, 'marker': None, 'limit': None}
    self.baremetal_mock.port.list.assert_called_with(**kwargs)
    collist = ('UUID', 'Address', 'Created At', 'Extra', 'Node UUID', 'Local Link Connection', 'Portgroup UUID', 'PXE boot enabled', 'Physical Network', 'Updated At', 'Internal Info', 'Is Smart NIC port')
    self.assertEqual(collist, columns)
    datalist = ((baremetal_fakes.baremetal_port_uuid, baremetal_fakes.baremetal_port_address, '', oscutils.format_dict(baremetal_fakes.baremetal_port_extra), baremetal_fakes.baremetal_uuid, '', '', '', '', '', '', ''),)
    self.assertEqual(datalist, tuple(data))
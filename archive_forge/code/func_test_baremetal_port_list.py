import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_port_list(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'marker': None, 'limit': None}
    self.baremetal_mock.port.list.assert_called_with(**kwargs)
    collist = ('UUID', 'Address')
    self.assertEqual(collist, columns)
    datalist = ((baremetal_fakes.baremetal_port_uuid, baremetal_fakes.baremetal_port_address),)
    self.assertEqual(datalist, tuple(data))
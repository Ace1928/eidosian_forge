import copy
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_chassis
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_chassis_list_no_options(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'marker': None, 'limit': None}
    self.baremetal_mock.chassis.list.assert_called_with(**kwargs)
    collist = ('UUID', 'Description')
    self.assertEqual(collist, columns)
    datalist = ((baremetal_fakes.baremetal_chassis_uuid, baremetal_fakes.baremetal_chassis_description),)
    self.assertEqual(datalist, tuple(data))
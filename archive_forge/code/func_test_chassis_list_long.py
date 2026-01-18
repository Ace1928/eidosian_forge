import copy
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_chassis
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_chassis_list_long(self):
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'detail': True, 'marker': None, 'limit': None}
    self.baremetal_mock.chassis.list.assert_called_with(**kwargs)
    collist = ('UUID', 'Description', 'Created At', 'Updated At', 'Extra')
    self.assertEqual(collist, columns)
    datalist = ((baremetal_fakes.baremetal_chassis_uuid, baremetal_fakes.baremetal_chassis_description, '', '', baremetal_fakes.baremetal_chassis_extra),)
    self.assertEqual(datalist, tuple(data))
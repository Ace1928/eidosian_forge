import copy
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_chassis
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_chassis_show(self):
    arglist = [baremetal_fakes.baremetal_chassis_uuid]
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    args = [baremetal_fakes.baremetal_chassis_uuid]
    self.baremetal_mock.chassis.get.assert_called_with(*args, fields=None)
    collist = ('description', 'extra', 'uuid')
    self.assertEqual(collist, columns)
    self.assertNotIn('nodes', columns)
    datalist = (baremetal_fakes.baremetal_chassis_description, baremetal_fakes.baremetal_chassis_extra, baremetal_fakes.baremetal_chassis_uuid)
    self.assertEqual(datalist, tuple(data))
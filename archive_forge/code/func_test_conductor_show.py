import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_conductor
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_conductor_show(self):
    arglist = ['xxxx.xxxx']
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    args = ['xxxx.xxxx']
    self.baremetal_mock.conductor.get.assert_called_with(*args, fields=None)
    collist = ('alive', 'conductor_group', 'drivers', 'hostname')
    self.assertEqual(collist, columns)
    datalist = (baremetal_fakes.baremetal_alive, baremetal_fakes.baremetal_conductor_group, baremetal_fakes.baremetal_drivers, baremetal_fakes.baremetal_hostname)
    self.assertEqual(datalist, tuple(data))
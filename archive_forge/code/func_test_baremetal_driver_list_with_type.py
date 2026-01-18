import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_driver
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_driver_list_with_type(self):
    arglist = ['--type', baremetal_fakes.baremetal_driver_type]
    verifylist = [('type', baremetal_fakes.baremetal_driver_type)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    collist = ('Supported driver(s)', 'Active host(s)')
    self.assertEqual(collist, tuple(columns))
    datalist = ((baremetal_fakes.baremetal_driver_name, ', '.join(baremetal_fakes.baremetal_driver_hosts)),)
    self.assertEqual(datalist, tuple(data))
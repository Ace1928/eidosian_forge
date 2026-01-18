import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_driver
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_driver_show_fields_multiple(self):
    arglist = ['fakedrivername', '--fields', 'name', '--fields', 'hosts', 'type']
    verifylist = [('driver', 'fakedrivername'), ('fields', [['name'], ['hosts', 'type']])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    args = ['fakedrivername']
    fields = ['name', 'hosts', 'type']
    self.baremetal_mock.driver.get.assert_called_with(*args, fields=fields)
    self.assertFalse(self.baremetal_mock.driver.properties.called)
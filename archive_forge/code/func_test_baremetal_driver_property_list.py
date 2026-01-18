import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_driver
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_driver_property_list(self):
    arglist = ['fakedrivername']
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.baremetal_mock.driver.properties.assert_called_with(*arglist)
    collist = ['Property', 'Description']
    self.assertEqual(collist, columns)
    expected_data = [('property1', 'description1'), ('property2', 'description2')]
    self.assertEqual(expected_data, data)
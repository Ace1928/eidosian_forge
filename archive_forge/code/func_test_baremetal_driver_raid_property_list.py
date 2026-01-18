import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_driver
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_driver_raid_property_list(self):
    arglist = ['fakedrivername']
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.baremetal_mock.driver.raid_logical_disk_properties.assert_called_with(*arglist)
    collist = ('Property', 'Description')
    self.assertEqual(collist, tuple(columns))
    expected_data = [('RAIDProperty1', 'driver_raid_property1'), ('RAIDProperty2', 'driver_raid_property2')]
    self.assertEqual(expected_data, data)
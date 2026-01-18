import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_conductor
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_conductor_list_long(self):
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'detail': True, 'marker': None, 'limit': None}
    self.baremetal_mock.conductor.list.assert_called_with(**kwargs)
    collist = ['Hostname', 'Conductor Group', 'Alive', 'Drivers', 'Created At', 'Updated At']
    self.assertEqual(tuple(collist), columns)
    fake_values = {'Hostname': baremetal_fakes.baremetal_hostname, 'Conductor Group': baremetal_fakes.baremetal_conductor_group, 'Alive': baremetal_fakes.baremetal_alive, 'Drivers': baremetal_fakes.baremetal_drivers}
    values = tuple((fake_values.get(name, '') for name in collist))
    self.assertEqual((values,), tuple(data))
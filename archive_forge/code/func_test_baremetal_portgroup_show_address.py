import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_show_address(self):
    arglist = ['--address', baremetal_fakes.baremetal_portgroup_address]
    verifylist = [('address', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    args = {baremetal_fakes.baremetal_portgroup_address}
    self.baremetal_mock.portgroup.get_by_address.assert_called_with(*args, fields=None)
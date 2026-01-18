import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_set_mode_int(self):
    new_portgroup_mode = '4'
    arglist = [baremetal_fakes.baremetal_portgroup_uuid, '--mode', new_portgroup_mode]
    verifylist = [('portgroup', baremetal_fakes.baremetal_portgroup_uuid), ('mode', new_portgroup_mode)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.portgroup.update.assert_called_once_with(baremetal_fakes.baremetal_portgroup_uuid, [{'path': '/mode', 'value': new_portgroup_mode, 'op': 'add'}])
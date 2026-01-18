import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_set_name(self):
    new_portgroup_name = 'New-Portgroup-name'
    arglist = [baremetal_fakes.baremetal_portgroup_uuid, '--name', new_portgroup_name]
    verifylist = [('portgroup', baremetal_fakes.baremetal_portgroup_uuid), ('name', new_portgroup_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.portgroup.update.assert_called_once_with(baremetal_fakes.baremetal_portgroup_uuid, [{'path': '/name', 'value': new_portgroup_name, 'op': 'add'}])
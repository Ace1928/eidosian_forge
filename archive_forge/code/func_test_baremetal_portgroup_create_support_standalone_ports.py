import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_create_support_standalone_ports(self):
    arglist = ['--address', baremetal_fakes.baremetal_portgroup_address, '--node', baremetal_fakes.baremetal_uuid, '--support-standalone-ports']
    verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('address', baremetal_fakes.baremetal_portgroup_address), ('support_standalone_ports', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    args = {'address': baremetal_fakes.baremetal_portgroup_address, 'node_uuid': baremetal_fakes.baremetal_uuid, 'standalone_ports_supported': True}
    self.baremetal_mock.portgroup.create.assert_called_once_with(**args)
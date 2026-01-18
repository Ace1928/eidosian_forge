import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_create_name_extras(self):
    arglist = ['--address', baremetal_fakes.baremetal_portgroup_address, '--node', baremetal_fakes.baremetal_uuid, '--name', baremetal_fakes.baremetal_portgroup_name, '--extra', 'key1=value1', '--extra', 'key2=value2']
    verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('address', baremetal_fakes.baremetal_portgroup_address), ('name', baremetal_fakes.baremetal_portgroup_name), ('extra', ['key1=value1', 'key2=value2'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    args = {'address': baremetal_fakes.baremetal_portgroup_address, 'node_uuid': baremetal_fakes.baremetal_uuid, 'name': baremetal_fakes.baremetal_portgroup_name, 'extra': baremetal_fakes.baremetal_portgroup_extra}
    self.baremetal_mock.portgroup.create.assert_called_once_with(**args)
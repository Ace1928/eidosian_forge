import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_port_set_portgroup_uuid(self):
    new_portgroup_uuid = '1111-111111-1111'
    arglist = [baremetal_fakes.baremetal_port_uuid, '--port-group', new_portgroup_uuid]
    verifylist = [('port', baremetal_fakes.baremetal_port_uuid), ('portgroup_uuid', new_portgroup_uuid)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.port.update.assert_called_once_with(baremetal_fakes.baremetal_port_uuid, [{'path': '/portgroup_uuid', 'value': new_portgroup_uuid, 'op': 'add'}])
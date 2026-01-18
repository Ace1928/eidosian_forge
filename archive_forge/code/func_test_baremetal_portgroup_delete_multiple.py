import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_delete_multiple(self):
    arglist = [baremetal_fakes.baremetal_portgroup_uuid, baremetal_fakes.baremetal_portgroup_name]
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    args = [baremetal_fakes.baremetal_portgroup_uuid, baremetal_fakes.baremetal_portgroup_name]
    self.baremetal_mock.portgroup.delete.assert_has_calls([mock.call(x) for x in args])
    self.assertEqual(2, self.baremetal_mock.portgroup.delete.call_count)
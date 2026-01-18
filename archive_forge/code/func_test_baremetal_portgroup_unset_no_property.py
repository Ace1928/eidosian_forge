import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_unset_no_property(self):
    uuid = baremetal_fakes.baremetal_portgroup_uuid
    arglist = [uuid]
    verifylist = [('portgroup', uuid)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.assertFalse(self.baremetal_mock.portgroup.update.called)
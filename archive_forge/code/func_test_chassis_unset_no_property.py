import copy
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_chassis
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_chassis_unset_no_property(self):
    uuid = baremetal_fakes.baremetal_chassis_uuid
    arglist = [uuid]
    verifylist = [('chassis', uuid)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.assertFalse(self.baremetal_mock.chassis.update.called)
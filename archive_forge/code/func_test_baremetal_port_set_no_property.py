import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_port_set_no_property(self):
    arglist = [baremetal_fakes.baremetal_port_uuid]
    verifylist = [('port', baremetal_fakes.baremetal_port_uuid)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.assertFalse(self.baremetal_mock.port.update.called)
import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_allocation
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_allocation_unset_name(self):
    arglist = [baremetal_fakes.baremetal_uuid, '--name']
    verifylist = [('allocation', baremetal_fakes.baremetal_uuid), ('name', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.allocation.update.assert_called_once_with(baremetal_fakes.baremetal_uuid, [{'path': '/name', 'op': 'remove'}])
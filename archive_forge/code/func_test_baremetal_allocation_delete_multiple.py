import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_allocation
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_allocation_delete_multiple(self):
    arglist = [baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_name]
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.allocation.delete.assert_has_calls([mock.call(x) for x in arglist])
    self.assertEqual(2, self.baremetal_mock.allocation.delete.call_count)
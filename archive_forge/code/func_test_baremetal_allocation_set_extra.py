import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_allocation
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_allocation_set_extra(self):
    extra_value = 'foo=bar'
    arglist = [baremetal_fakes.baremetal_uuid, '--extra', extra_value]
    verifylist = [('allocation', baremetal_fakes.baremetal_uuid), ('extra', [extra_value])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.allocation.update.assert_called_once_with(baremetal_fakes.baremetal_uuid, [{'path': '/extra/foo', 'value': 'bar', 'op': 'add'}])
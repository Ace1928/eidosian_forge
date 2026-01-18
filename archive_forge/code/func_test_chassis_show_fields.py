import copy
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_chassis
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_chassis_show_fields(self):
    uuid = baremetal_fakes.baremetal_chassis_uuid
    arglist = [uuid, '--fields', 'uuid', 'description']
    verifylist = [('chassis', uuid), ('fields', [['uuid', 'description']])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    args = [uuid]
    fields = ['uuid', 'description']
    self.baremetal_mock.chassis.get.assert_called_with(*args, fields=fields)
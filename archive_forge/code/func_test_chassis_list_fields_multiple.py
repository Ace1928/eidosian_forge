import copy
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_chassis
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_chassis_list_fields_multiple(self):
    arglist = ['--fields', 'uuid', 'description', '--fields', 'extra']
    verifylist = [('fields', [['uuid', 'description'], ['extra']])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    kwargs = {'marker': None, 'limit': None, 'detail': False, 'fields': ('uuid', 'description', 'extra')}
    self.baremetal_mock.chassis.list.assert_called_with(**kwargs)
import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_allocation
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_allocation_list_fields_multiple(self):
    arglist = ['--fields', 'uuid', 'node_uuid', '--fields', 'extra']
    verifylist = [('fields', [['uuid', 'node_uuid'], ['extra']])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    kwargs = {'marker': None, 'limit': None, 'fields': ('uuid', 'node_uuid', 'extra')}
    self.baremetal_mock.allocation.list.assert_called_once_with(**kwargs)
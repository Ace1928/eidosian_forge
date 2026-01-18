import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_conductor
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_conductor_list_fields_multiple(self):
    arglist = ['--fields', 'hostname', 'alive', '--fields', 'conductor_group']
    verifylist = [('fields', [['hostname', 'alive'], ['conductor_group']])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    kwargs = {'marker': None, 'limit': None, 'detail': False, 'fields': ('hostname', 'alive', 'conductor_group')}
    self.baremetal_mock.conductor.list.assert_called_with(**kwargs)
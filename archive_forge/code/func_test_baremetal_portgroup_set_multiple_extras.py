import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_set_multiple_extras(self):
    arglist = ['portgroup', '--extra', 'key1=val1', '--extra', 'key2=val2']
    verifylist = [('portgroup', 'portgroup'), ('extra', ['key1=val1', 'key2=val2'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.portgroup.update.assert_called_once_with('portgroup', [{'path': '/extra/key1', 'value': 'val1', 'op': 'add'}, {'path': '/extra/key2', 'value': 'val2', 'op': 'add'}])
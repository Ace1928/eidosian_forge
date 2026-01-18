import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_unset_multiple_properties(self):
    arglist = ['portgroup', '--property', 'key1', '--property', 'key2']
    verifylist = [('portgroup', 'portgroup'), ('properties', ['key1', 'key2'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.portgroup.update.assert_called_once_with('portgroup', [{'path': '/properties/key1', 'op': 'remove'}, {'path': '/properties/key2', 'op': 'remove'}])
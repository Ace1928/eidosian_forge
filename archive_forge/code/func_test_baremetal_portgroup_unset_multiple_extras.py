import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_unset_multiple_extras(self):
    arglist = ['portgroup', '--extra', 'key1', '--extra', 'key2']
    verifylist = [('portgroup', 'portgroup'), ('extra', ['key1', 'key2'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.portgroup.update.assert_called_once_with('portgroup', [{'path': '/extra/key1', 'op': 'remove'}, {'path': '/extra/key2', 'op': 'remove'}])
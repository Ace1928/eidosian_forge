from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair_group
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_delete_multiple_port_pair_groups_with_exception(self):
    client = self.app.client_manager.network
    target1 = 'target'
    arglist = [target1]
    verifylist = [('port_pair_group', [target1])]
    client.find_sfc_port_pair_group.side_effect = [target1, exceptions.CommandError]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    msg = '1 of 2 port pair group(s) failed to delete.'
    with testtools.ExpectedException(exceptions.CommandError) as e:
        self.cmd.take_action(parsed_args)
        self.assertEqual(msg, str(e))
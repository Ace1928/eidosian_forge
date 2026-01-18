import testtools
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
def test_delete_multiple_with_exception(self):
    target1 = 'target'
    arglist = [target1]
    verifylist = [(self.res, [target1])]
    self.networkclient.find_firewall_group.side_effect = [target1, exceptions.CommandError]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    resource_name = self.res.replace('_', ' ')
    msg = '1 of 2 %s(s) failed to delete.' % resource_name
    with testtools.ExpectedException(exceptions.CommandError) as e:
        self.cmd.take_action(parsed_args)
        self.assertEqual(msg, str(e))
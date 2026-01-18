import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7policy
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_l7policy_delete_failure(self):
    arglist = ['unknown_policy']
    verifylist = [('l7policy', 'unknown_policy')]
    self.api_mock.l7policy_list.return_value = {'l7policies': []}
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertNotCalled(self.api_mock.l7policy_delete)
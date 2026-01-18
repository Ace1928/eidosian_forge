import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7policy
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_l7policy_unset_none(self):
    self.api_mock.l7policy_set.reset_mock()
    arglist = [self._l7po.id]
    verifylist = list(zip(self.PARAMETERS, [False] * len(self.PARAMETERS)))
    verifylist = [('l7policy', self._l7po.id)] + verifylist
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.l7policy_set.assert_not_called()
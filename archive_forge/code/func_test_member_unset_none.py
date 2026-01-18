import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import member
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_member_unset_none(self):
    self.api_mock.pool_set.reset_mock()
    arglist = [self._mem.pool_id, self._mem.id]
    verifylist = list(zip(self.PARAMETERS, [False] * len(self.PARAMETERS)))
    verifylist = [('member', self._mem.id)] + verifylist
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.member_set.assert_not_called()
import copy
from unittest import mock
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7rule
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_l7rule_show(self):
    arglist = [self._l7po.id, self._l7ru.id]
    verifylist = [('l7policy', self._l7po.id), ('l7rule', self._l7ru.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.l7rule_show.assert_called_with(l7rule_id=self._l7ru.id, l7policy_id=self._l7po.id)
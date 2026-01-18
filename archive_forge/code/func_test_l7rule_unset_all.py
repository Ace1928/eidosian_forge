import copy
from unittest import mock
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7rule
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_l7rule_unset_all(self):
    self.api_mock.l7rule_set.reset_mock()
    ref_body = {'rule': {x: None for x in self.PARAMETERS}}
    arglist = [self._l7po.id, self._l7ru.id]
    for ref_param in self.PARAMETERS:
        arg_param = ref_param.replace('_', '-') if '_' in ref_param else ref_param
        arglist.append('--%s' % arg_param)
    verifylist = list(zip(self.PARAMETERS, [True] * len(self.PARAMETERS)))
    verifylist = [('l7rule_id', self._l7ru.id)] + verifylist
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.l7rule_set.assert_called_once_with(l7policy_id=self._l7po.id, l7rule_id=self._l7ru.id, json=ref_body)
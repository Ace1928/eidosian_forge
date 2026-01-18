import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import member
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_member_unset_all(self):
    self.api_mock.pool_set.reset_mock()
    ref_body = {'member': {x: None for x in self.PARAMETERS}}
    arglist = [self._mem.pool_id, self._mem.id]
    for ref_param in self.PARAMETERS:
        arg_param = ref_param.replace('_', '-') if '_' in ref_param else ref_param
        arglist.append('--%s' % arg_param)
    verifylist = list(zip(self.PARAMETERS, [True] * len(self.PARAMETERS)))
    verifylist = [('member', self._mem.id)] + verifylist
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.member_set.assert_called_once_with(member_id=self._mem.id, pool_id=self._mem.pool_id, json=ref_body)
import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import listener
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_listener_unset_all(self):
    self.api_mock.listener_set.reset_mock()
    ref_body = {'listener': {x: None for x in self.PARAMETERS}}
    arglist = [self._listener.id]
    for ref_param in self.PARAMETERS:
        if ref_param == 'default_pool_id':
            ref_param = 'default_pool'
        arg_param = ref_param.replace('_', '-') if '_' in ref_param else ref_param
        arglist.append('--%s' % arg_param)
    verifylist = list(zip(self.PARAMETERS, [True] * len(self.PARAMETERS)))
    verifylist = [('listener', self._listener.id)] + verifylist
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.listener_set.assert_called_once_with(self._listener.id, json=ref_body)
import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7policy
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_l7policy_attrs')
def test_l7policy_create_with_tag(self, mock_attrs):
    mock_attrs.return_value = {'listener_id': self._l7po.listener_id, 'name': self._l7po.name, 'action': 'REDIRECT_TO_POOL', 'redirect_pool_id': self._l7po.redirect_pool_id, 'tags': ['foo']}
    arglist = ['mock_li_id', '--name', self._l7po.name, '--action', 'REDIRECT_TO_POOL'.lower(), '--redirect-pool', self._l7po.redirect_pool_id, '--tag', 'foo']
    verifylist = [('listener', 'mock_li_id'), ('name', self._l7po.name), ('action', 'REDIRECT_TO_POOL'), ('redirect_pool', self._l7po.redirect_pool_id), ('tags', ['foo'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.l7policy_create.assert_called_with(json={'l7policy': {'listener_id': self._l7po.listener_id, 'name': self._l7po.name, 'action': 'REDIRECT_TO_POOL', 'redirect_pool_id': self._l7po.redirect_pool_id, 'tags': ['foo']}})
import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7policy
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_l7policy_attrs')
def test_l7policy_list_by_listener(self, mock_l7policy_attrs):
    mock_l7policy_attrs.return_value = {'listener_id': self._l7po.listener_id}
    arglist = ['--listener', 'mock_li_id']
    verifylist = [('listener', 'mock_li_id')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.api_mock.l7policy_list.assert_called_with(listener_id=self._l7po.listener_id)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, tuple(data))
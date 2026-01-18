import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import member
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_member_attrs')
def test_member_set_tag_no_tag(self, mock_attrs):
    self.api_mock.member_show.return_value = {'tags': ['foo']}
    mock_attrs.return_value = {'pool_id': self._mem.pool_id, 'member_id': self._mem.id, 'tags': ['bar']}
    arglist = [self._mem.pool_id, self._mem.id, '--tag', 'bar', '--no-tag']
    verifylist = [('pool', self._mem.pool_id), ('member', self._mem.id), ('tags', ['bar'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.member_set.assert_called_once_with(pool_id=self._mem.pool_id, member_id=self._mem.id, json={'member': {'tags': ['bar']}})
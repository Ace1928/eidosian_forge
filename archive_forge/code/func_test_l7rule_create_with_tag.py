import copy
from unittest import mock
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7rule
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_l7rule_attrs')
def test_l7rule_create_with_tag(self, mock_attrs):
    mock_attrs.return_value = {'l7policy_id': self._l7po.id, 'compare-type': 'ENDS_WITH', 'value': '.example.com', 'type': 'HOST_NAME', 'tags': ['foo']}
    arglist = [self._l7po.id, '--compare-type', 'ENDS_WITH', '--value', '.example.com', '--type', 'HOST_NAME'.lower(), '--tag', 'foo']
    verifylist = [('l7policy', self._l7po.id), ('compare_type', 'ENDS_WITH'), ('value', '.example.com'), ('type', 'HOST_NAME'), ('tags', ['foo'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.l7rule_create.assert_called_with(l7policy_id=self._l7po.id, json={'rule': {'compare-type': 'ENDS_WITH', 'value': '.example.com', 'type': 'HOST_NAME', 'tags': ['foo']}})
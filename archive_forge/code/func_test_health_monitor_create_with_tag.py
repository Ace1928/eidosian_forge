import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import health_monitor
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_health_monitor_attrs')
def test_health_monitor_create_with_tag(self, mock_attrs):
    mock_attrs.return_value = self.hm_info
    arglist = ['mock_pool_id', '--name', self._hm.name, '--delay', str(self._hm.delay), '--timeout', str(self._hm.timeout), '--max-retries', str(self._hm.max_retries), '--type', self._hm.type.lower(), '--tag', 'foo']
    verifylist = [('pool', 'mock_pool_id'), ('name', self._hm.name), ('delay', str(self._hm.delay)), ('timeout', str(self._hm.timeout)), ('max_retries', self._hm.max_retries), ('type', self._hm.type), ('tags', ['foo'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.health_monitor_create.assert_called_with(json={'healthmonitor': self.hm_info})
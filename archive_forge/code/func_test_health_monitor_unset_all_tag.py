import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import health_monitor
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_health_monitor_unset_all_tag(self):
    self.api_mock.health_monitor_set.reset_mock()
    self.api_mock.health_monitor_show.return_value = {'tags': ['foo', 'bar']}
    arglist = [self._hm.id, '--all-tag']
    verifylist = [('health_monitor', self._hm.id), ('all_tag', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.health_monitor_set.assert_called_once_with(self._hm.id, json={'healthmonitor': {'tags': []}})
import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import health_monitor
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_health_monitor_set(self):
    arglist = [self._hm.id, '--name', 'new_name', '--http-version', str(self._hm.http_version), '--domain-name', self._hm.domain_name]
    verifylist = [('health_monitor', self._hm.id), ('name', 'new_name'), ('http_version', self._hm.http_version), ('domain_name', self._hm.domain_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.health_monitor_set.assert_called_with(self._hm.id, json={'healthmonitor': {'name': 'new_name', 'http_version': self._hm.http_version, 'domain_name': self._hm.domain_name}})
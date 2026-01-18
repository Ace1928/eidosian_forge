import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import listener
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_listener_set_suppressed(self):
    arglist = [self._listener.id, '--name', 'foo']
    verifylist = [('name', 'foo')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.assertNotIn('hsts_preload', parsed_args)
    self.assertNotIn('hsts_include_subdomain', parsed_args)
    self.assertNotIn('hsts_max_age', parsed_args)
    self.api_mock.listener_set.assert_called_with(self._listener.id, json={'listener': {'name': 'foo'}})
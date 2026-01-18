import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import listener
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_listener_set_tag_no_tag(self):
    self.api_mock.listener_show.return_value = {'tags': ['foo']}
    arglist = [self._listener.id, '--tag', 'bar', '--no-tag']
    verifylist = [('listener', self._listener.id), ('tags', ['bar'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.listener_set.assert_called_once_with(self._listener.id, json={'listener': {'tags': ['bar']}})
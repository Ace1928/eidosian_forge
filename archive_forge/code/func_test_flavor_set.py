import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import flavor
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_flavor_set(self):
    arglist = [self._flavor.id, '--name', 'new_name', '--description', 'new_desc']
    verifylist = [('flavor', self._flavor.id), ('name', 'new_name'), ('description', 'new_desc')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.flavor_set.assert_called_with(self._flavor.id, json={'flavor': {'name': 'new_name', 'description': 'new_desc'}})
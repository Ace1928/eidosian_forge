import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import availabilityzone
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_availabilityzone_set(self):
    arglist = [self._availabilityzone.name, '--description', 'new_desc']
    verifylist = [('availabilityzone', self._availabilityzone.name), ('description', 'new_desc')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.availabilityzone_set.assert_called_with(self._availabilityzone.name, json={'availability_zone': {'description': 'new_desc'}})
import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import availabilityzone
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_availabilityzone_delete_failure(self):
    arglist = ['unknown_availabilityzone']
    verifylist = [('availabilityzone', 'unknown_availabilityzone')]
    self.api_mock.availabilityzone_list.return_value = {'availability_zones': []}
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertNotCalled(self.api_mock.availabilityzone_delete)
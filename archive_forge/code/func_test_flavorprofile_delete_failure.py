import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import flavorprofile
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_flavorprofile_delete_failure(self):
    arglist = ['unknown_flavorprofile']
    verifylist = [('flavorprofile', 'unknown_flavorprofile')]
    self.api_mock.flavorprofile_list.return_value = {'flavorprofiles': []}
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertNotCalled(self.api_mock.flavorprofile_delete)
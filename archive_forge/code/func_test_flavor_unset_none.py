import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import flavor
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_flavor_unset_none(self):
    self.api_mock.flavor_set.reset_mock()
    arglist = [self._flavor.id]
    verifylist = list(zip(self.PARAMETERS, [False] * len(self.PARAMETERS)))
    verifylist = [('flavor', self._flavor.id)] + verifylist
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.flavor_set.assert_not_called()
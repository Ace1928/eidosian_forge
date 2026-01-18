import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import availabilityzone
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_availabilityzone_list_with_options(self):
    arglist = ['--name', 'availabilityzone1']
    verifylist = [('name', 'availabilityzone1')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.api_mock.availabilityzone_list.assert_called_with(name='availabilityzone1')
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, tuple(data))
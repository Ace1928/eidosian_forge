import random
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_list_with_tag_options(self):
    arglist = ['--tags', 'red,blue', '--any-tags', 'red,green', '--not-tags', 'orange,yellow', '--not-any-tags', 'black,white']
    verifylist = [('tags', ['red', 'blue']), ('any_tags', ['red', 'green']), ('not_tags', ['orange', 'yellow']), ('not_any_tags', ['black', 'white'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.networks.assert_called_once_with(**{'tags': 'red,blue', 'any_tags': 'red,green', 'not_tags': 'orange,yellow', 'not_any_tags': 'black,white'})
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))
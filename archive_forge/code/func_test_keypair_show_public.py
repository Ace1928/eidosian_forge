import copy
from unittest import mock
from unittest.mock import call
import uuid
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import keypair
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_keypair_show_public(self):
    self.keypair = compute_fakes.create_one_keypair()
    self.compute_sdk_client.find_keypair.return_value = self.keypair
    arglist = ['--public-key', self.keypair.name]
    verifylist = [('public_key', True), ('name', self.keypair.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual({}, columns)
    self.assertEqual({}, data)
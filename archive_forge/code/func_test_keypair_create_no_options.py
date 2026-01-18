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
@mock.patch.object(keypair, '_generate_keypair', return_value=keypair.Keypair('private', 'public'))
def test_keypair_create_no_options(self, mock_generate):
    arglist = [self.keypair.name]
    verifylist = [('name', self.keypair.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.create_keypair.assert_called_with(name=self.keypair.name, public_key=mock_generate.return_value.public_key)
    self.assertEqual({}, columns)
    self.assertEqual({}, data)
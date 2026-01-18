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
@mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
def test_key_pair_create_with_user(self, sm_mock, mock_generate):
    arglist = ['--user', identity_fakes.user_name, self.keypair.name]
    verifylist = [('user', identity_fakes.user_name), ('name', self.keypair.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.create_keypair.assert_called_with(name=self.keypair.name, user_id=identity_fakes.user_id, public_key=mock_generate.return_value.public_key)
    self.assertEqual({}, columns)
    self.assertEqual({}, data)
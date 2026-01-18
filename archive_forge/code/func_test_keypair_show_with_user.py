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
@mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
def test_keypair_show_with_user(self, sm_mock):
    self.keypair = compute_fakes.create_one_keypair()
    self.compute_sdk_client.find_keypair.return_value = self.keypair
    self.data = (self.keypair.created_at, self.keypair.fingerprint, self.keypair.id, self.keypair.is_deleted, self.keypair.name, self.keypair.private_key, self.keypair.type, self.keypair.user_id)
    arglist = ['--user', identity_fakes.user_name, self.keypair.name]
    verifylist = [('user', identity_fakes.user_name), ('name', self.keypair.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.users_mock.get.assert_called_with(identity_fakes.user_name)
    self.compute_sdk_client.find_keypair.assert_called_with(self.keypair.name, ignore_missing=False, user_id=identity_fakes.user_id)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)
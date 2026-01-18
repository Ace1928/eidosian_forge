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
def test_keypair_delete_with_user(self, sm_mock):
    arglist = ['--user', identity_fakes.user_name, self.keypairs[0].name]
    verifylist = [('user', identity_fakes.user_name), ('name', [self.keypairs[0].name])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    ret = self.cmd.take_action(parsed_args)
    self.assertIsNone(ret)
    self.compute_sdk_client.delete_keypair.assert_called_with(self.keypairs[0].name, user_id=identity_fakes.user_id, ignore_missing=False)
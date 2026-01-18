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
@mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
def test_keypair_show_with_user_pre_v210(self, sm_mock):
    self.keypair = compute_fakes.create_one_keypair()
    arglist = ['--user', identity_fakes.user_name, self.keypair.name]
    verifylist = [('user', identity_fakes.user_name), ('name', self.keypair.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-compute-api-version 2.10 or greater is required', str(ex))
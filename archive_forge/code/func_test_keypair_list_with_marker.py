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
@mock.patch.object(sdk_utils, 'supports_microversion', new=mock.Mock(return_value=True))
def test_keypair_list_with_marker(self):
    arglist = ['--marker', 'test_kp']
    verifylist = [('marker', 'test_kp')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.compute_sdk_client.keypairs.assert_called_with(marker='test_kp')
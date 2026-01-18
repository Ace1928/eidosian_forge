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
def test_keypair_create_with_key_type_pre_v22(self, sm_mock):
    for key_type in ['x509', 'ssh']:
        arglist = ['--public-key', self.keypair.public_key, self.keypair.name, '--type', 'ssh']
        verifylist = [('public_key', self.keypair.public_key), ('name', self.keypair.name), ('type', 'ssh')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('io.open') as mock_open:
            mock_open.return_value = mock.MagicMock()
            m_file = mock_open.return_value.__enter__.return_value
            m_file.read.return_value = 'dummy'
            ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.2 or greater is required', str(ex))
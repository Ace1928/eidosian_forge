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
def test_keypair_create_private_key(self, mock_generate):
    tmp_pk_file = '/tmp/kp-file-' + uuid.uuid4().hex
    arglist = ['--private-key', tmp_pk_file, self.keypair.name]
    verifylist = [('private_key', tmp_pk_file), ('name', self.keypair.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('io.open') as mock_open:
        mock_open.return_value = mock.MagicMock()
        m_file = mock_open.return_value.__enter__.return_value
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.create_keypair.assert_called_with(name=self.keypair.name, public_key=mock_generate.return_value.public_key)
        mock_open.assert_called_once_with(tmp_pk_file, 'w+')
        m_file.write.assert_called_once_with(mock_generate.return_value.private_key)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)
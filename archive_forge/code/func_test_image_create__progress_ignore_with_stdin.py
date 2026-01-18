import copy
import io
import tempfile
from unittest import mock
from cinderclient import api_versions
from openstack import exceptions as sdk_exceptions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.image.v2 import image as _image
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
@mock.patch('openstackclient.image.v2.image.get_data_from_stdin')
def test_image_create__progress_ignore_with_stdin(self, mock_get_data_from_stdin):
    fake_stdin = io.BytesIO(b'some fake data')
    mock_get_data_from_stdin.return_value = fake_stdin
    arglist = ['--progress', self.new_image.name]
    verifylist = [('progress', True), ('name', self.new_image.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.image_client.create_image.assert_called_with(name=self.new_image.name, allow_duplicates=True, container_format=_image.DEFAULT_CONTAINER_FORMAT, disk_format=_image.DEFAULT_DISK_FORMAT, data=fake_stdin, validate_checksum=False)
    self.assertEqual(self.expected_columns, columns)
    self.assertCountEqual(self.expected_data, data)
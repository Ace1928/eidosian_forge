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
@mock.patch('osc_lib.utils.find_resource')
@mock.patch('openstackclient.image.v2.image.get_data_from_stdin')
def test_image_create_from_volume_fail(self, mock_get_data_f, mock_get_vol):
    fake_vol_id = 'fake-volume-id'
    mock_get_data_f.return_value = None

    class FakeVolume:
        id = fake_vol_id
    mock_get_vol.return_value = FakeVolume()
    arglist = ['--volume', fake_vol_id, self.new_image.name, '--public']
    verifylist = [('name', self.new_image.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
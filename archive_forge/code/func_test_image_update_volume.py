import copy
from unittest import mock
from osc_lib.cli import format_columns
from openstackclient.image.v1 import image
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.image.v1 import fakes as image_fakes
def test_image_update_volume(self):
    volumes_mock = self.volume_client.volumes
    volumes_mock.reset_mock()
    volumes_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy({'id': 'vol1', 'name': 'volly'}), loaded=True)
    response = {'id': 'volume_id', 'updated_at': 'updated_at', 'status': 'uploading', 'display_description': 'desc', 'size': 'size', 'volume_type': 'volume_type', 'container_format': image.DEFAULT_CONTAINER_FORMAT, 'disk_format': image.DEFAULT_DISK_FORMAT, 'image': self._image.name}
    full_response = {'os-volume_upload_image': response}
    volumes_mock.upload_to_image.return_value = (201, full_response)
    arglist = ['--volume', 'volly', '--name', 'updated_image', self._image.name]
    verifylist = [('private', False), ('protected', False), ('public', False), ('unprotected', False), ('volume', 'volly'), ('force', False), ('name', 'updated_image'), ('image', self._image.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    volumes_mock.upload_to_image.assert_called_with('vol1', False, self._image.name, '', '')
    self.image_client.update_image.assert_called_with(self._image.id, name='updated_image', volume='volly')
    self.assertIsNone(result)
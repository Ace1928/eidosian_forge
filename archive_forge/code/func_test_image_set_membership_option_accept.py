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
def test_image_set_membership_option_accept(self):
    membership = image_fakes.create_one_image_member(attrs={'image_id': '0f41529e-7c12-4de8-be2d-181abb825b3c', 'member_id': self.project.id})
    self.image_client.update_member.return_value = membership
    arglist = ['--accept', self._image.id]
    verifylist = [('membership', 'accepted'), ('image', self._image.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.image_client.update_member.assert_called_once_with(image=self._image.id, member=self.app.client_manager.auth_ref.project_id, status='accepted')
    self.image_client.update_image.assert_called_with(self._image.id)
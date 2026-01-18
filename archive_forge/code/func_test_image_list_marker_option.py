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
def test_image_list_marker_option(self, fr_mock):
    self.image_client.find_image = mock.Mock(return_value=self._image)
    arglist = ['--marker', 'graven']
    verifylist = [('marker', 'graven')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.image_client.images.assert_called_with(marker=self._image.id)
    self.image_client.find_image.assert_called_with('graven', ignore_missing=False)
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
@mock.patch('sys.stdin', side_effect=[None])
def test_image_create_import(self, raw_input):
    arglist = ['--import', self.new_image.name]
    verifylist = [('name', self.new_image.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.image_client.create_image.assert_called_with(name=self.new_image.name, allow_duplicates=True, container_format=_image.DEFAULT_CONTAINER_FORMAT, disk_format=_image.DEFAULT_DISK_FORMAT, use_import=True)
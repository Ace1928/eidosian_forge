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
def test_image_create_with_unexist_project(self):
    self.project_mock.get.side_effect = exceptions.NotFound(None)
    self.project_mock.find.side_effect = exceptions.NotFound(None)
    arglist = ['--container-format', 'ovf', '--disk-format', 'ami', '--min-disk', '10', '--min-ram', '4', '--protected', '--private', '--project', 'unexist_owner', 'graven']
    verifylist = [('container_format', 'ovf'), ('disk_format', 'ami'), ('min_disk', 10), ('min_ram', 4), ('is_protected', True), ('visibility', 'private'), ('project', 'unexist_owner'), ('name', 'graven')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
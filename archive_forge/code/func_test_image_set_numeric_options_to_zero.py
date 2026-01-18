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
def test_image_set_numeric_options_to_zero(self):
    arglist = ['--min-disk', '0', '--min-ram', '0', 'graven']
    verifylist = [('min_disk', 0), ('min_ram', 0), ('image', 'graven')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'min_disk': 0, 'min_ram': 0}
    self.image_client.update_image.assert_called_with(self._image.id, **kwargs)
    self.assertIsNone(result)
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
def test_image_unset_mixed_option(self):
    arglist = ['--tag', 'test', '--property', 'hw_rng_model', '--property', 'prop', self.image.id]
    verifylist = [('tags', ['test']), ('properties', ['hw_rng_model', 'prop']), ('image', self.image.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.image_client.update_image.assert_called_with(self.image, properties={'prop2': 'fake'})
    self.image_client.remove_tag.assert_called_with(self.image.id, 'test')
    self.assertIsNone(result)
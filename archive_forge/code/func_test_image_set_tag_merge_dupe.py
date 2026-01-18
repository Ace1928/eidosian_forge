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
def test_image_set_tag_merge_dupe(self):
    old_image = self._image
    old_image['tags'] = ['old1', 'new2']
    self.image_client.find_image.return_value = old_image
    arglist = ['--tag', 'old1', 'graven']
    verifylist = [('tags', ['old1']), ('image', 'graven')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'tags': ['new2', 'old1']}
    a, k = self.image_client.update_image.call_args
    self.assertEqual(self._image.id, a[0])
    self.assertIn('tags', k)
    self.assertEqual(set(kwargs['tags']), set(k['tags']))
    self.assertIsNone(result)
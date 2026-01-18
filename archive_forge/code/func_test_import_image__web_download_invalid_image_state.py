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
def test_import_image__web_download_invalid_image_state(self):
    self.image.status = 'uploading'
    arglist = [self.image.name, '--method', 'web-download', '--uri', 'https://example.com/']
    verifylist = [('image', self.image.name), ('import_method', 'web-download'), ('uri', 'https://example.com/')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn("The 'web-download' import method can only be used with an image in status 'queued'", str(exc))
    self.image_client.import_image.assert_not_called()
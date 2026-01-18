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
def test_image_list_long_option(self):
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.image_client.images.assert_called_with()
    collist = ('ID', 'Name', 'Disk Format', 'Container Format', 'Size', 'Checksum', 'Status', 'Visibility', 'Protected', 'Project', 'Tags')
    self.assertEqual(collist, columns)
    datalist = ((self._image.id, self._image.name, None, None, None, None, None, self._image.visibility, self._image.is_protected, self._image.owner_id, format_columns.ListColumn(self._image.tags)),)
    self.assertCountEqual(datalist, tuple(data))
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
def test_get_data_from_stdin__interactive(self):
    fd = io.BytesIO(b'some initial binary data: \x00\x01')
    with mock.patch('sys.stdin') as stdin:
        stdin.return_value = fd
        test_fd = _image.get_data_from_stdin()
        self.assertIsNone(test_fd)
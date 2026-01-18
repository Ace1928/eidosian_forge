import errno
import hashlib
import testtools
from unittest import mock
import ddt
from glanceclient.common import utils as common_utils
from glanceclient import exc
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import images
def test_data_upload_w_size(self):
    image_data = 'CCC'
    image_id = '606b0e88-7c5a-4d54-b5bb-046105d4de6f'
    self.controller.upload(image_id, image_data, image_size=3)
    expect = [('PUT', '/v2/images/%s/file' % image_id, {'Content-Type': 'application/octet-stream'}, image_data)]
    self.assertEqual(expect, self.api.calls)
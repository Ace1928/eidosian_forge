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
def test_image_import_web_download(self):
    uri = 'http://example.com/image.qcow'
    data = [('method', {'name': 'web-download', 'uri': uri})]
    image_id = '606b0e88-7c5a-4d54-b5bb-046105d4de6f'
    self.controller.image_import(image_id, 'web-download', uri)
    expect = [('POST', '/v2/images/%s/import' % image_id, {}, data)]
    self.assertEqual(expect, self.api.calls)
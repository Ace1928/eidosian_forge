import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_image_meta_from_headers_encoding(self):
    value = u'niño'
    fields = {'x-image-meta-name': value}
    headers = self.mgr._image_meta_from_headers(fields)
    self.assertEqual(value, headers['name'])
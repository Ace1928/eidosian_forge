import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_image_list_all_tenants(self):
    images = self.mgr.list(is_public=None, page_size=5)
    image_list = list(images)
    self.assertEqual('A', image_list[0].owner)
    self.assertEqual('a', image_list[0].id)
    self.assertEqual('B', image_list[1].owner)
    self.assertEqual('b', image_list[1].id)
    self.assertEqual('B', image_list[2].owner)
    self.assertEqual('b2', image_list[2].id)
    self.assertRaises(AttributeError, lambda: image_list[3].owner)
    self.assertEqual('c', image_list[3].id)
    self.assertEqual(4, len(image_list))
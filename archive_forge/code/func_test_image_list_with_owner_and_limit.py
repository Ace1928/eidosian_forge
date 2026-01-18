import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_image_list_with_owner_and_limit(self):
    images = self.mgr.list(owner='B', page_size=5, limit=1)
    image_list = list(images)
    self.assertEqual('B', image_list[0].owner)
    self.assertEqual('b', image_list[0].id)
    self.assertEqual(1, len(image_list))
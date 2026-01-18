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
def test_list_images_paginated_with_limit(self):
    images = self.controller.list(limit=3, page_size=2)
    self.assertEqual('3a4560a1-e585-443e-9b39-553b46ec92d1', images[0].id)
    self.assertEqual('image-1', images[0].name)
    self.assertEqual('6f99bf80-2ee6-47cf-acfe-1f1fabb7e810', images[1].id)
    self.assertEqual('image-2', images[1].name)
    self.assertEqual('3f99bf80-2ee6-47cf-acfe-1f1fabb7e811', images[2].id)
    self.assertEqual('image-3', images[2].name)
    self.assertEqual(3, len(images))
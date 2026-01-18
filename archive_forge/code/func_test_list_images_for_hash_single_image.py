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
def test_list_images_for_hash_single_image(self):
    fake_id = '3a4560a1-e585-443e-9b39-553b46ec92d1'
    filters = {'filters': {'os_hash_value': _HASHVAL}}
    images = self.controller.list(**filters)
    self.assertEqual(1, len(images))
    self.assertEqual('%s' % fake_id, images[0].id)
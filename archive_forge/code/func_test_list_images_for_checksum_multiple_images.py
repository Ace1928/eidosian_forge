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
def test_list_images_for_checksum_multiple_images(self):
    fake_id1 = '2a4560b2-e585-443e-9b39-553b46ec92d1'
    fake_id2 = '6f99bf80-2ee6-47cf-acfe-1f1fabb7e810'
    filters = {'filters': {'checksum': _CHKSUM1}}
    images = self.controller.list(**filters)
    self.assertEqual(2, len(images))
    self.assertEqual('%s' % fake_id1, images[0].id)
    self.assertEqual('%s' % fake_id2, images[1].id)
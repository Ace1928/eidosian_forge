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
def test_list_images_sort_dirs_fewer_than_keys(self):
    sort_key = ['name', 'id', 'created_at']
    sort_dir = ['desc', 'asc']
    self.assertRaises(exc.HTTPBadRequest, self.controller.list, sort_key=sort_key, sort_dir=sort_dir)
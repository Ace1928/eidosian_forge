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
def test_list_images_filters_encoding(self):
    filters = {'owner': u'niño'}
    try:
        self.controller.list(filters=filters)
    except KeyError:
        pass
    self.assertEqual(b'ni\xc3\xb1o', filters['owner'])
import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_get_encoding(self):
    image = self.mgr.get('3')
    self.assertEqual(u'niño', image.name)
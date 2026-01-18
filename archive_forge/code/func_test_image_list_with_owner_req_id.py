import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_image_list_with_owner_req_id(self):
    fields = {'owner': 'A', 'return_req_id': []}
    images = self.mgr.list(**fields)
    next(images)
    self.assertEqual(['req-1234'], fields['return_req_id'])
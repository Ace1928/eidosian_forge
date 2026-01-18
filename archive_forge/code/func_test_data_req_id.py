import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_data_req_id(self):
    params = {'do_checksum': False, 'return_req_id': []}
    ''.join([b for b in self.mgr.data('4', **params)])
    expect_req_id = ['req-1234']
    self.assertEqual(expect_req_id, params['return_req_id'])
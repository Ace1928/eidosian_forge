import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def test_containers_put_archive(self):
    response = self.mgr.put_archive(CONTAINER1['id'], path, data)
    expect = [('POST', '/v1/containers/%s/put_archive?%s' % (CONTAINER1['id'], parse.urlencode({'path': path})), {'Content-Length': '0'}, {'data': zun_utils.encode_file_data(data)})]
    self.assertEqual(expect, self.api.calls)
    self.assertTrue(response)
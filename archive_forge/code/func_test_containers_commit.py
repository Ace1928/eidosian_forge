import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def test_containers_commit(self):
    containers = self.mgr.commit(CONTAINER1['id'], repo, tag)
    expect = [('POST', '/v1/containers/%s/commit?%s' % (CONTAINER1['id'], parse.urlencode({'repository': repo, 'tag': tag})), {'Content-Length': '0'}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertIsNone(containers)
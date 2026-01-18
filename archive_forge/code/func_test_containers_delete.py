import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def test_containers_delete(self):
    containers = self.mgr.delete(CONTAINER1['id'], force=force_delete1)
    expect = [('DELETE', '/v1/containers/%s?force=%s' % (CONTAINER1['id'], force_delete1), {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertIsNone(containers)
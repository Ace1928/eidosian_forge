import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def test_container_show_by_id(self):
    container = self.mgr.get(CONTAINER1['id'])
    expect = [('GET', '/v1/containers/%s' % CONTAINER1['id'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(CONTAINER1['name'], container.name)
    self.assertEqual(CONTAINER1['uuid'], container.uuid)
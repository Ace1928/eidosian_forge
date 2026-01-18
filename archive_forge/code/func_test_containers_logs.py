import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def test_containers_logs(self):
    containers = self.mgr.logs(CONTAINER1['id'], stdout=True, stderr=True, timestamps=False, tail='all', since=None)
    expect = [('GET', '/v1/containers/%s/logs?%s' % (CONTAINER1['id'], parse.urlencode({'stdout': True, 'stderr': True, 'timestamps': False, 'tail': 'all', 'since': None})), {'Content-Length': '0'}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertIsNone(containers)
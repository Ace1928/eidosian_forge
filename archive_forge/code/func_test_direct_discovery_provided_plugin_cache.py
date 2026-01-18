import abc
import collections
import urllib
import uuid
from keystoneauth1 import _utils
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_direct_discovery_provided_plugin_cache(self):
    resps = [{'json': self.TEST_DISCOVERY}, {'status_code': 500}]
    self.requests_mock.get(self.TEST_COMPUTE_ADMIN, resps)
    sa = session.Session()
    sb = session.Session()
    discovery_cache = {}
    expected_url = self.TEST_COMPUTE_ADMIN + '/v2.0'
    for sess in (sa, sb):
        disc = discover.get_discovery(sess, self.TEST_COMPUTE_ADMIN, cache=discovery_cache)
        url = disc.url_for(('2', '0'))
        self.assertEqual(expected_url, url)
    self.assertIn(self.TEST_COMPUTE_ADMIN, discovery_cache.keys())
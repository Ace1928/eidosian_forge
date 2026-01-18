import copy
import queue
from unittest import mock
from keystoneauth1 import session
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_get_bypass_cache(self):
    key = self._get_key(4)
    resp = copy.deepcopy(self.response)
    resp.body = {'foo': 'bar'}
    self.cloud._api_cache_keys.add(key)
    self.cloud._cache.set(key, resp)
    self.cloud._cache_expirations['srv.fake'] = 5
    self.sot._get(self.Res, '4', skip_cache=True)
    self.session.request.assert_called()
    self.assertEqual(dict(), self.response.body)
    self.assertNotIn(key, self.cloud._api_cache_keys)
    self.assertEqual('NoValue', type(self.cloud._cache.get(key)).__name__)
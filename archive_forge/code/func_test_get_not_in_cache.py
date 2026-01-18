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
def test_get_not_in_cache(self):
    self.cloud._cache_expirations['srv.fake'] = 5
    self.sot._get(self.Res, '1')
    self.session.request.assert_called_with('fake/1', 'GET', connect_retries=mock.ANY, raise_exc=mock.ANY, global_request_id=mock.ANY, microversion=mock.ANY, params=mock.ANY, endpoint_filter=mock.ANY, headers=mock.ANY, rate_semaphore=mock.ANY)
    self.assertIn(self._get_key(1), self.cloud._api_cache_keys)
import datetime
import os
import time
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystoneauth1 import session
import oslo_cache
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import pbr.version
import testresources
from testtools import matchers
import webob
import webob.dec
from keystonemiddleware import auth_token
from keystonemiddleware.auth_token import _base
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit import client_fixtures
def test_caching_token_on_verify(self):
    self.middleware._token_cache._env_cache_name = 'cache'
    cache = _cache._FakeClient()
    self.middleware._token_cache.initialize(env={'cache': cache})
    orig_cache_set = cache.set
    cache.set = mock.Mock(side_effect=orig_cache_set)
    token = self.token_dict['uuid_token_default']
    self.call_middleware(headers={'X-Auth-Token': token})
    self.assertThat(1, matchers.Equals(cache.set.call_count))
    self.call_middleware(headers={'X-Auth-Token': token})
    self.assertThat(1, matchers.Equals(cache.set.call_count))
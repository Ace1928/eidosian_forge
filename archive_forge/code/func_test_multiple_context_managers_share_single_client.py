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
def test_multiple_context_managers_share_single_client(self):
    self.set_middleware()
    token_cache = self.middleware._token_cache
    env = {}
    token_cache.initialize(env)
    caches = []
    with token_cache._cache_pool.reserve() as cache:
        caches.append(cache)
    with token_cache._cache_pool.reserve() as cache:
        caches.append(cache)
    self.assertIs(caches[0], caches[1])
    self.assertEqual(set(caches), set(token_cache._cache_pool))
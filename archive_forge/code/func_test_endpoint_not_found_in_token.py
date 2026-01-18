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
def test_endpoint_not_found_in_token(self):
    token = ENDPOINT_NOT_FOUND_TOKEN
    self.set_middleware()
    self.middleware._token_cache.initialize({})
    with mock.patch.object(self.middleware._identity_server, 'invalidate', new=mock.Mock()):
        self.assertRaises(ksa_exceptions.EndpointNotFound, self.middleware.fetch_token, token)
        self.assertTrue(self.middleware._identity_server.invalidate.called)
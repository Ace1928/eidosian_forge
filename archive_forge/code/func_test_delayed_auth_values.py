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
def test_delayed_auth_values(self):
    conf = {'www_authenticate_uri': 'http://local.test'}
    status = '401 Unauthorized'
    middleware = self.create_simple_middleware(status=status, conf=conf)
    self.assertFalse(middleware._delay_auth_decision)
    for v in ('True', '1', 'on', 'yes'):
        conf = {'delay_auth_decision': v, 'www_authenticate_uri': 'http://local.test'}
        middleware = self.create_simple_middleware(status=status, conf=conf)
        self.assertTrue(middleware._delay_auth_decision)
    for v in ('False', '0', 'no'):
        conf = {'delay_auth_decision': v, 'www_authenticate_uri': 'http://local.test'}
        middleware = self.create_simple_middleware(status=status, conf=conf)
        self.assertFalse(middleware._delay_auth_decision)
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
def test_micro_version(self):
    self._assert_auth_version('v3', (3, 0))
    self._assert_auth_version('v3.0', (3, 0))
    self._assert_auth_version('v3.1', (3, 0))
    self._assert_auth_version('v3.2', (3, 0))
    self._assert_auth_version('v3.9', (3, 0))
    self._assert_auth_version('v3.3.1', (3, 0))
    self._assert_auth_version('v3.3.5', (3, 0))
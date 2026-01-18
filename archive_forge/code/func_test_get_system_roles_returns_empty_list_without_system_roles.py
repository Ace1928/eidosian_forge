import copy
import datetime
import fixtures
import itertools
import operator
import re
from unittest import mock
from urllib import parse
import uuid
from cryptography.hazmat.primitives.serialization import Encoding
import freezegun
import http.client
from oslo_serialization import jsonutils as json
from oslo_utils import fixture
from oslo_utils import timeutils
from testtools import matchers
from testtools import testcase
from keystone import auth
from keystone.auth.plugins import totp
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
def test_get_system_roles_returns_empty_list_without_system_roles(self):
    unscoped_request = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'])
    r = self.post('/auth/tokens', body=unscoped_request)
    unscoped_token = r.headers.get('X-Subject-Token')
    self.assertValidUnscopedTokenResponse(r)
    response = self.get('/auth/system', token=unscoped_token)
    self.assertEqual(response.json_body['system'], [])
    self.head('/auth/system', token=unscoped_token, expected_status=http.client.OK)
    project_scoped_request = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project_id)
    r = self.post('/auth/tokens', body=project_scoped_request)
    project_scoped_token = r.headers.get('X-Subject-Token')
    self.assertValidProjectScopedTokenResponse(r)
    response = self.get('/auth/system', token=project_scoped_token)
    self.assertEqual(response.json_body['system'], [])
    self.head('/auth/system', token=project_scoped_token, expected_status=http.client.OK)
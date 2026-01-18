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
def test_create_implied_role_shows_in_v3_system_token(self):
    self.config_fixture.config(group='token')
    PROVIDERS.assignment_api.create_system_grant_for_user(self.user['id'], self.role['id'])
    token_id = self.get_system_scoped_token()
    r = self.get('/auth/tokens', headers={'X-Subject-Token': token_id})
    token_roles = r.result['token']['roles']
    prior = token_roles[0]['id']
    self._create_implied_role(prior)
    r = self.get('/auth/tokens', headers={'X-Subject-Token': token_id})
    token_roles = r.result['token']['roles']
    self.assertEqual(2, len(token_roles))
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
def test_authenticate_without_trust_dict_returns_bad_request(self):
    token = self.v3_create_token(self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'])).headers.get('X-Subject-Token')
    auth_data = {'auth': {'identity': {'methods': ['token'], 'token': {'id': token}}, 'scope': {'OS-TRUST:trust': ''}}}
    self.admin_request(method='POST', path='/v3/auth/tokens', body=auth_data, expected_status=http.client.BAD_REQUEST)
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
def test_create_token_for_user_without_password_fails(self):
    user = unit.new_user_ref(domain_id=self.domain['id'])
    del user['password']
    user = PROVIDERS.identity_api.create_user(user)
    auth_data = self.build_authentication_request(user_id=user['id'], password='password')
    self.v3_create_token(auth_data, expected_status=http.client.UNAUTHORIZED)
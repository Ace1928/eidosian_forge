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
def test_delete_trust_revokes_tokens(self):
    ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user['id'], project_id=self.project_id, impersonation=False, expires=dict(minutes=1), role_ids=[self.role_id])
    r = self.post('/OS-TRUST/trusts', body={'trust': ref})
    trust = self.assertValidTrustResponse(r)
    trust_id = trust['id']
    auth_data = self.build_authentication_request(user_id=self.trustee_user['id'], password=self.trustee_user['password'], trust_id=trust_id)
    r = self.v3_create_token(auth_data)
    self.assertValidProjectScopedTokenResponse(r, self.trustee_user)
    trust_token = r.headers['X-Subject-Token']
    self.delete('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust_id})
    headers = {'X-Subject-Token': trust_token}
    self.head('/auth/tokens', headers=headers, expected_status=http.client.NOT_FOUND)
    self.assertTrustTokensRevoked(trust_id)
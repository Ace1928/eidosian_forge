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
def test_application_credential_cannot_scope(self):
    app_cred = self._make_app_cred()
    app_cred_ref = self.app_cred_api.create_application_credential(app_cred)
    new_project_ref = unit.new_project_ref(domain_id=self.domain_id)
    new_project = PROVIDERS.resource_api.create_project(new_project_ref['id'], new_project_ref)
    PROVIDERS.assignment_api.add_role_to_user_and_project(self.user['id'], new_project['id'], self.role_id)
    password_auth = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=new_project['id'])
    password_response = self.v3_create_token(password_auth)
    self.assertValidProjectScopedTokenResponse(password_response)
    app_cred_auth = self.build_authentication_request(app_cred_id=app_cred_ref['id'], secret=app_cred_ref['secret'], project_id=new_project['id'])
    self.v3_create_token(app_cred_auth, expected_status=http.client.UNAUTHORIZED)
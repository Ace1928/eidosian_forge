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
def test_create_project_token_with_same_domain_and_project_name(self):
    """Authenticate to a project with the same name as its domain."""
    domain = unit.new_project_ref(is_domain=True)
    domain = PROVIDERS.resource_api.create_project(domain['id'], domain)
    project = unit.new_project_ref(domain_id=domain['id'], name=domain['name'])
    PROVIDERS.resource_api.create_project(project['id'], project)
    role_member = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role_member['id'], role_member)
    PROVIDERS.assignment_api.add_role_to_user_and_project(self.user['id'], project['id'], role_member['id'])
    auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_name=project['name'], project_domain_name=domain['name'])
    r = self.v3_create_token(auth_data)
    self.assertEqual(project['id'], r.result['token']['project']['id'])
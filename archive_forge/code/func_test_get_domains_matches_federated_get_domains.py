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
def test_get_domains_matches_federated_get_domains(self):
    ref = unit.new_domain_ref()
    r = self.post('/domains', body={'domain': ref})
    unauthorized_domain_id = r.json['domain']['id']
    ref = unit.new_domain_ref()
    r = self.post('/domains', body={'domain': ref})
    authorized_domain_id = r.json['domain']['id']
    path = '/domains/%(domain_id)s/users/%(user_id)s/roles/%(role_id)s' % {'domain_id': authorized_domain_id, 'user_id': self.user_id, 'role_id': self.role_id}
    self.put(path, expected_status=http.client.NO_CONTENT)
    r = self.get('/auth/domains', expected_status=http.client.OK)
    self.assertThat(r.json['domains'], matchers.HasLength(1))
    self.assertEqual(authorized_domain_id, r.json['domains'][0]['id'])
    self.assertNotEqual(unauthorized_domain_id, r.json['domains'][0]['id'])
    r = self.get('/OS-FEDERATION/domains', expected_status=http.client.OK)
    self.assertThat(r.json['domains'], matchers.HasLength(1))
    self.assertEqual(authorized_domain_id, r.json['domains'][0]['id'])
    self.assertNotEqual(unauthorized_domain_id, r.json['domains'][0]['id'])
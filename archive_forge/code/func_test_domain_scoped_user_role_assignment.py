import copy
import os
import random
import re
import subprocess
from testtools import matchers
from unittest import mock
import uuid
import fixtures
import flask
import http.client
from lxml import etree
from oslo_serialization import jsonutils
from oslo_utils import importutils
import saml2
from saml2 import saml
from saml2 import sigver
import urllib
from keystone.api._shared import authentication
from keystone.api import auth as auth_api
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import render_token
import keystone.conf
from keystone import exception
from keystone.federation import idp as keystone_idp
from keystone.models import token_model
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import core
from keystone.tests.unit import federation_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_domain_scoped_user_role_assignment(self):
    domain_ref = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain_ref['id'], domain_ref)
    role_ref = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
    user_id, unscoped_token = self._authenticate_via_saml()
    v3_scope_request = self._scope_request(unscoped_token, 'domain', domain_ref['id'])
    r = self.v3_create_token(v3_scope_request, expected_status=http.client.UNAUTHORIZED)
    PROVIDERS.assignment_api.create_grant(role_ref['id'], user_id=user_id, domain_id=domain_ref['id'])
    r = self.v3_create_token(v3_scope_request, expected_status=http.client.CREATED)
    self.assertIsNotNone(r.headers.get('X-Subject-Token'))
    token_resp = r.result['token']
    self.assertIn('domain', token_resp)
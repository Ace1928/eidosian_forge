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
def test_default_domain_scoped_token(self):
    self.config_fixture.config(group='token', cache_on_issue=False)
    token = self._issue_unscoped_token()
    PROVIDERS.assignment_api.create_grant(self.role_admin['id'], user_id=token.user_id, domain_id=CONF.identity.default_domain_id)
    auth_request = {'auth': {'identity': {'methods': ['token'], 'token': {'id': token.id}}, 'scope': {'domain': {'id': CONF.identity.default_domain_id}}}}
    r = self.v3_create_token(auth_request)
    domain_scoped_token_id = r.headers.get('X-Subject-Token')
    headers = {'X-Subject-Token': domain_scoped_token_id}
    self.get('/auth/tokens', token=domain_scoped_token_id, headers=headers)
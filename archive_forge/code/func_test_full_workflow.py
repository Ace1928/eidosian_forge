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
def test_full_workflow(self):
    """Test 'standard' workflow for granting access tokens.

        * Issue unscoped token
        * List available projects based on groups
        * Scope token to one of available projects

        """
    r = self._issue_unscoped_token()
    token_resp = render_token.render_token_response_from_model(r)['token']
    self.assertListEqual(['saml2'], r.methods)
    self.assertValidMappedUser(token_resp)
    employee_unscoped_token_id = r.id
    r = self.get('/auth/projects', token=employee_unscoped_token_id)
    projects = r.result['projects']
    random_project = random.randint(0, len(projects) - 1)
    project = projects[random_project]
    v3_scope_request = self._scope_request(employee_unscoped_token_id, 'project', project['id'])
    r = self.v3_create_token(v3_scope_request)
    token_resp = r.result['token']
    self.assertIn('token', token_resp['methods'])
    self.assertIn('saml2', token_resp['methods'])
    self._check_project_scoped_token_attributes(token_resp, project['id'])
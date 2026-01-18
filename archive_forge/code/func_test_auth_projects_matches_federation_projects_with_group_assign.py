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
def test_auth_projects_matches_federation_projects_with_group_assign(self):
    domain_id = CONF.identity.default_domain_id
    project_ref = unit.new_project_ref(domain_id=domain_id)
    PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
    role_ref = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
    group_ref = unit.new_group_ref(domain_id=domain_id)
    group_ref = PROVIDERS.identity_api.create_group(group_ref)
    user_id, unscoped_token = self._authenticate_via_saml()
    PROVIDERS.assignment_api.create_grant(role_ref['id'], group_id=group_ref['id'], project_id=project_ref['id'], domain_id=domain_id)
    PROVIDERS.identity_api.add_user_to_group(user_id=user_id, group_id=group_ref['id'])
    r = self.get('/auth/projects', token=unscoped_token)
    auth_projects = r.result['projects']
    r = self.get('/OS-FEDERATION/projects', token=unscoped_token)
    fed_projects = r.result['projects']
    self.assertCountEqual(auth_projects, fed_projects)
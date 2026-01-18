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
def test_list_head_projects_for_user_duplicates(self):
    role_ref = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
    user_id, unscoped_token = self._authenticate_via_saml()
    r = self.get('/OS-FEDERATION/projects', token=unscoped_token)
    group_projects = r.result['projects']
    project_from_group = group_projects[0]
    self.head('/OS-FEDERATION/projects', token=unscoped_token, expected_status=http.client.OK)
    PROVIDERS.assignment_api.add_role_to_user_and_project(user_id, project_from_group['id'], role_ref['id'])
    r = self.get('/OS-FEDERATION/projects', token=unscoped_token)
    user_projects = r.result['projects']
    user_project_ids = []
    for project in user_projects:
        self.assertNotIn(project['id'], user_project_ids)
        user_project_ids.append(project['id'])
    r = self.get('/auth/projects', token=unscoped_token)
    user_projects = r.result['projects']
    user_project_ids = []
    for project in user_projects:
        self.assertNotIn(project['id'], user_project_ids)
        user_project_ids.append(project['id'])
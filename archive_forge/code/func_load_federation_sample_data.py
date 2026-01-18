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
def load_federation_sample_data(self):
    """Inject additional data."""
    self.domainA = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(self.domainA['id'], self.domainA)
    self.domainB = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(self.domainB['id'], self.domainB)
    self.domainC = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(self.domainC['id'], self.domainC)
    self.domainD = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(self.domainD['id'], self.domainD)
    self.proj_employees = unit.new_project_ref(domain_id=self.domainA['id'])
    PROVIDERS.resource_api.create_project(self.proj_employees['id'], self.proj_employees)
    self.proj_customers = unit.new_project_ref(domain_id=self.domainA['id'])
    PROVIDERS.resource_api.create_project(self.proj_customers['id'], self.proj_customers)
    self.project_all = unit.new_project_ref(domain_id=self.domainA['id'])
    PROVIDERS.resource_api.create_project(self.project_all['id'], self.project_all)
    self.project_inherited = unit.new_project_ref(domain_id=self.domainD['id'])
    PROVIDERS.resource_api.create_project(self.project_inherited['id'], self.project_inherited)
    self.group_employees = unit.new_group_ref(domain_id=self.domainA['id'])
    self.group_employees = PROVIDERS.identity_api.create_group(self.group_employees)
    self.group_customers = unit.new_group_ref(domain_id=self.domainA['id'])
    self.group_customers = PROVIDERS.identity_api.create_group(self.group_customers)
    self.group_admins = unit.new_group_ref(domain_id=self.domainA['id'])
    self.group_admins = PROVIDERS.identity_api.create_group(self.group_admins)
    self.role_employee = unit.new_role_ref()
    PROVIDERS.role_api.create_role(self.role_employee['id'], self.role_employee)
    self.role_customer = unit.new_role_ref()
    PROVIDERS.role_api.create_role(self.role_customer['id'], self.role_customer)
    self.role_admin = unit.new_role_ref()
    PROVIDERS.role_api.create_role(self.role_admin['id'], self.role_admin)
    PROVIDERS.assignment_api.create_grant(self.role_employee['id'], group_id=self.group_employees['id'], project_id=self.proj_employees['id'])
    PROVIDERS.assignment_api.create_grant(self.role_employee['id'], group_id=self.group_employees['id'], project_id=self.project_all['id'])
    PROVIDERS.assignment_api.create_grant(self.role_customer['id'], group_id=self.group_customers['id'], project_id=self.proj_customers['id'])
    PROVIDERS.assignment_api.create_grant(self.role_admin['id'], group_id=self.group_admins['id'], project_id=self.proj_customers['id'])
    PROVIDERS.assignment_api.create_grant(self.role_admin['id'], group_id=self.group_admins['id'], project_id=self.proj_employees['id'])
    PROVIDERS.assignment_api.create_grant(self.role_admin['id'], group_id=self.group_admins['id'], project_id=self.project_all['id'])
    PROVIDERS.assignment_api.create_grant(self.role_customer['id'], group_id=self.group_customers['id'], domain_id=self.domainA['id'])
    PROVIDERS.assignment_api.create_grant(self.role_customer['id'], group_id=self.group_customers['id'], domain_id=self.domainD['id'], inherited_to_projects=True)
    PROVIDERS.assignment_api.create_grant(self.role_employee['id'], group_id=self.group_employees['id'], domain_id=self.domainA['id'])
    PROVIDERS.assignment_api.create_grant(self.role_employee['id'], group_id=self.group_employees['id'], domain_id=self.domainB['id'])
    PROVIDERS.assignment_api.create_grant(self.role_admin['id'], group_id=self.group_admins['id'], domain_id=self.domainA['id'])
    PROVIDERS.assignment_api.create_grant(self.role_admin['id'], group_id=self.group_admins['id'], domain_id=self.domainB['id'])
    PROVIDERS.assignment_api.create_grant(self.role_admin['id'], group_id=self.group_admins['id'], domain_id=self.domainC['id'])
    self.rules = {'rules': [{'local': [{'group': {'id': self.group_employees['id']}}, {'user': {'name': '{0}', 'id': '{1}'}}], 'remote': [{'type': 'UserName'}, {'type': 'Email'}, {'type': 'orgPersonType', 'any_one_of': ['Employee']}]}, {'local': [{'group': {'id': self.group_employees['id']}}, {'user': {'name': '{0}', 'id': '{1}'}}], 'remote': [{'type': self.ASSERTION_PREFIX + 'UserName'}, {'type': self.ASSERTION_PREFIX + 'Email'}, {'type': self.ASSERTION_PREFIX + 'orgPersonType', 'any_one_of': ['SuperEmployee']}]}, {'local': [{'group': {'id': self.group_customers['id']}}, {'user': {'name': '{0}', 'id': '{1}'}}], 'remote': [{'type': 'UserName'}, {'type': 'Email'}, {'type': 'orgPersonType', 'any_one_of': ['Customer']}]}, {'local': [{'group': {'id': self.group_admins['id']}}, {'group': {'id': self.group_employees['id']}}, {'group': {'id': self.group_customers['id']}}, {'user': {'name': '{0}', 'id': '{1}'}}], 'remote': [{'type': 'UserName'}, {'type': 'Email'}, {'type': 'orgPersonType', 'any_one_of': ['Admin', 'Chief']}]}, {'local': [{'group': {'id': uuid.uuid4().hex}}, {'group': {'id': self.group_customers['id']}}, {'user': {'name': '{0}', 'id': '{1}'}}], 'remote': [{'type': 'UserName'}, {'type': 'Email'}, {'type': 'FirstName', 'any_one_of': ['Jill']}, {'type': 'LastName', 'any_one_of': ['Smith']}]}, {'local': [{'group': {'id': 'this_group_no_longer_exists'}}, {'user': {'name': '{0}', 'id': '{1}'}}], 'remote': [{'type': 'UserName'}, {'type': 'Email'}, {'type': 'Email', 'any_one_of': ['testacct@example.com']}, {'type': 'orgPersonType', 'any_one_of': ['Tester']}]}, {'local': [{'user': {'name': '{0}', 'id': '{1}'}}, {'group': {'name': self.group_customers['name'], 'domain': {'name': self.domainA['name']}}}], 'remote': [{'type': 'UserName'}, {'type': 'Email'}, {'type': 'orgPersonType', 'any_one_of': ['CEO', 'CTO']}]}, {'local': [{'user': {'name': '{0}', 'id': '{1}'}}, {'group': {'name': self.group_admins['name'], 'domain': {'id': self.domainA['id']}}}], 'remote': [{'type': 'UserName'}, {'type': 'Email'}, {'type': 'orgPersonType', 'any_one_of': ['Managers']}]}, {'local': [{'user': {'name': '{0}', 'id': '{1}'}}, {'group': {'name': 'NON_EXISTING', 'domain': {'id': self.domainA['id']}}}], 'remote': [{'type': 'UserName'}, {'type': 'Email'}, {'type': 'UserName', 'any_one_of': ['IamTester']}]}, {'local': [{'user': {'type': 'local', 'name': self.user['name'], 'domain': {'id': self.user['domain_id']}}}, {'group': {'id': self.group_customers['id']}}], 'remote': [{'type': 'UserType', 'any_one_of': ['random']}]}, {'local': [{'user': {'type': 'local', 'name': self.user['name'], 'domain': {'id': uuid.uuid4().hex}}}], 'remote': [{'type': 'Position', 'any_one_of': ['DirectorGeneral']}]}, {'local': [{'user': {'name': '{0}', 'id': '{1}'}}], 'remote': [{'type': 'UserName'}, {'type': 'Email'}, {'type': 'orgPersonType', 'any_one_of': ['NoGroupsOrg']}]}]}
    self.dummy_idp = self.idp_ref()
    PROVIDERS.federation_api.create_idp(self.dummy_idp['id'], self.dummy_idp)
    self.idp = self.idp_ref(id=self.IDP)
    PROVIDERS.federation_api.create_idp(self.idp['id'], self.idp)
    self.idp_with_remote = self.idp_ref(id=self.IDP_WITH_REMOTE)
    self.idp_with_remote['remote_ids'] = self.REMOTE_IDS
    PROVIDERS.federation_api.create_idp(self.idp_with_remote['id'], self.idp_with_remote)
    self.mapping = self.mapping_ref()
    PROVIDERS.federation_api.create_mapping(self.mapping['id'], self.mapping)
    self.proto_saml = self.proto_ref(mapping_id=self.mapping['id'])
    self.proto_saml['id'] = self.PROTOCOL
    PROVIDERS.federation_api.create_protocol(self.idp['id'], self.proto_saml['id'], self.proto_saml)
    PROVIDERS.federation_api.create_protocol(self.idp_with_remote['id'], self.proto_saml['id'], self.proto_saml)
    self.proto_dummy = self.proto_ref(mapping_id=self.mapping['id'])
    PROVIDERS.federation_api.create_protocol(self.dummy_idp['id'], self.proto_dummy['id'], self.proto_dummy)
    with self.make_request():
        self.tokens = {}
        VARIANTS = ('EMPLOYEE_ASSERTION', 'CUSTOMER_ASSERTION', 'ADMIN_ASSERTION')
        for variant in VARIANTS:
            self._inject_assertion(variant)
            r = authentication.authenticate_for_token(self.UNSCOPED_V3_SAML2_REQ)
            self.tokens[variant] = r.id
        self.TOKEN_SCOPE_PROJECT_FROM_NONEXISTENT_TOKEN = self._scope_request(uuid.uuid4().hex, 'project', self.proj_customers['id'])
        self.TOKEN_SCOPE_PROJECT_EMPLOYEE_FROM_EMPLOYEE = self._scope_request(self.tokens['EMPLOYEE_ASSERTION'], 'project', self.proj_employees['id'])
        self.TOKEN_SCOPE_PROJECT_EMPLOYEE_FROM_ADMIN = self._scope_request(self.tokens['ADMIN_ASSERTION'], 'project', self.proj_employees['id'])
        self.TOKEN_SCOPE_PROJECT_CUSTOMER_FROM_ADMIN = self._scope_request(self.tokens['ADMIN_ASSERTION'], 'project', self.proj_customers['id'])
        self.TOKEN_SCOPE_PROJECT_EMPLOYEE_FROM_CUSTOMER = self._scope_request(self.tokens['CUSTOMER_ASSERTION'], 'project', self.proj_employees['id'])
        self.TOKEN_SCOPE_PROJECT_INHERITED_FROM_CUSTOMER = self._scope_request(self.tokens['CUSTOMER_ASSERTION'], 'project', self.project_inherited['id'])
        self.TOKEN_SCOPE_DOMAIN_A_FROM_CUSTOMER = self._scope_request(self.tokens['CUSTOMER_ASSERTION'], 'domain', self.domainA['id'])
        self.TOKEN_SCOPE_DOMAIN_B_FROM_CUSTOMER = self._scope_request(self.tokens['CUSTOMER_ASSERTION'], 'domain', self.domainB['id'])
        self.TOKEN_SCOPE_DOMAIN_D_FROM_CUSTOMER = self._scope_request(self.tokens['CUSTOMER_ASSERTION'], 'domain', self.domainD['id'])
        self.TOKEN_SCOPE_DOMAIN_A_FROM_ADMIN = self._scope_request(self.tokens['ADMIN_ASSERTION'], 'domain', self.domainA['id'])
        self.TOKEN_SCOPE_DOMAIN_B_FROM_ADMIN = self._scope_request(self.tokens['ADMIN_ASSERTION'], 'domain', self.domainB['id'])
        self.TOKEN_SCOPE_DOMAIN_C_FROM_ADMIN = self._scope_request(self.tokens['ADMIN_ASSERTION'], 'domain', self.domainC['id'])
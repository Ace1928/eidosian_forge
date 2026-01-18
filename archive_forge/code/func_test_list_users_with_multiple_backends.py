import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_log import log
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import base as identity_base
from keystone.identity.backends import resource_options as options
from keystone.identity.backends import sql_model as model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_list_users_with_multiple_backends(self):
    """Call ``GET /users`` when multiple backends is enabled.

        In this scenario, the controller requires a domain to be specified
        either as a filter or by using a domain scoped token.

        """
    self.config_fixture.config(group='identity', domain_specific_drivers_enabled=True)
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    project = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project['id'], project)
    user = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
    PROVIDERS.assignment_api.create_grant(role_id=self.role_id, user_id=user['id'], domain_id=domain['id'])
    PROVIDERS.assignment_api.create_grant(role_id=self.role_id, user_id=user['id'], project_id=project['id'])
    dom_auth = self.build_authentication_request(user_id=user['id'], password=user['password'], domain_id=domain['id'])
    project_auth = self.build_authentication_request(user_id=user['id'], password=user['password'], project_id=project['id'])
    resource_url = '/users'
    r = self.get(resource_url, auth=dom_auth)
    self.assertValidUserListResponse(r, ref=user, resource_url=resource_url)
    resource_url = '/users'
    r = self.get(resource_url, auth=project_auth)
    self.assertValidUserListResponse(r, ref=user, resource_url=resource_url)
    resource_url = '/users?domain_id=%(domain_id)s' % {'domain_id': domain['id']}
    r = self.get(resource_url)
    self.assertValidUserListResponse(r, ref=user, resource_url=resource_url)
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
def test_create_user_without_domain(self):
    """Call ``POST /users`` without specifying domain.

        According to the identity-api specification, if you do not
        explicitly specific the domain_id in the entity, it should
        take the domain scope of the token as the domain_id.

        """
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    user = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
    PROVIDERS.assignment_api.create_grant(role_id=self.role_id, user_id=user['id'], domain_id=domain['id'])
    ref = unit.new_user_ref(domain_id=domain['id'])
    ref_nd = ref.copy()
    ref_nd.pop('domain_id')
    auth = self.build_authentication_request(user_id=user['id'], password=user['password'], domain_id=domain['id'])
    r = self.post('/users', body={'user': ref_nd}, auth=auth)
    self.assertValidUserResponse(r, ref)
    ref = unit.new_user_ref(domain_id=domain['id'])
    ref_nd = ref.copy()
    ref_nd.pop('domain_id')
    auth = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
    with mock.patch('oslo_log.versionutils.report_deprecated_feature') as mock_dep:
        r = self.post('/users', body={'user': ref_nd}, auth=auth)
        self.assertTrue(mock_dep.called)
    ref['domain_id'] = CONF.identity.default_domain_id
    return self.assertValidUserResponse(r, ref)
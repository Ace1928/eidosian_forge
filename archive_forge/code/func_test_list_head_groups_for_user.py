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
def test_list_head_groups_for_user(self):
    """Call ``GET & HEAD /users/{user_id}/groups``."""
    user1 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
    user2 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
    self.put('/groups/%(group_id)s/users/%(user_id)s' % {'group_id': self.group_id, 'user_id': user1['id']})
    auth = self.build_authentication_request(user_id=user1['id'], password=user1['password'])
    resource_url = '/users/%(user_id)s/groups' % {'user_id': user1['id']}
    r = self.get(resource_url, auth=auth)
    self.assertValidGroupListResponse(r, ref=self.group, resource_url=resource_url)
    self.head(resource_url, auth=auth, expected_status=http.client.OK)
    resource_url = '/users/%(user_id)s/groups' % {'user_id': user1['id']}
    r = self.get(resource_url)
    self.assertValidGroupListResponse(r, ref=self.group, resource_url=resource_url)
    self.head(resource_url, expected_status=http.client.OK)
    auth = self.build_authentication_request(user_id=user2['id'], password=user2['password'])
    resource_url = '/users/%(user_id)s/groups' % {'user_id': user1['id']}
    self.get(resource_url, auth=auth, expected_status=exception.ForbiddenAction.code)
    self.head(resource_url, auth=auth, expected_status=exception.ForbiddenAction.code)
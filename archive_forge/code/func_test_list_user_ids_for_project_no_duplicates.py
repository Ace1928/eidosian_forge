from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_user_ids_for_project_no_duplicates(self):
    user_ref = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user_ref = PROVIDERS.identity_api.create_user(user_ref)
    project_ref = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
    for i in range(2):
        role_ref = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
        PROVIDERS.assignment_api.add_role_to_user_and_project(user_id=user_ref['id'], project_id=project_ref['id'], role_id=role_ref['id'])
    user_ids = PROVIDERS.assignment_api.list_user_ids_for_project(project_ref['id'])
    self.assertEqual(1, len(user_ids))
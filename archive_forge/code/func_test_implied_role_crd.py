from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_implied_role_crd(self):
    prior_role_ref = unit.new_role_ref()
    PROVIDERS.role_api.create_role(prior_role_ref['id'], prior_role_ref)
    implied_role_ref = unit.new_role_ref()
    PROVIDERS.role_api.create_role(implied_role_ref['id'], implied_role_ref)
    PROVIDERS.role_api.create_implied_role(prior_role_ref['id'], implied_role_ref['id'])
    implied_role = PROVIDERS.role_api.get_implied_role(prior_role_ref['id'], implied_role_ref['id'])
    expected_implied_role_ref = {'prior_role_id': prior_role_ref['id'], 'implied_role_id': implied_role_ref['id']}
    self.assertLessEqual(expected_implied_role_ref.items(), implied_role.items())
    PROVIDERS.role_api.delete_implied_role(prior_role_ref['id'], implied_role_ref['id'])
    self.assertRaises(exception.ImpliedRoleNotFound, PROVIDERS.role_api.get_implied_role, uuid.uuid4().hex, uuid.uuid4().hex)
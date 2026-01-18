from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_crud_inherited_and_direct_assignment_for_group_on_project(self):
    group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
    group = PROVIDERS.identity_api.create_group(group)
    self._test_crud_inherited_and_direct_assignment(group_id=group['id'], project_id=self.project_baz['id'])
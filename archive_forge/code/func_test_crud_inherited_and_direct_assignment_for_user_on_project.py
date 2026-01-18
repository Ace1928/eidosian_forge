from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_crud_inherited_and_direct_assignment_for_user_on_project(self):
    self._test_crud_inherited_and_direct_assignment(user_id=self.user_foo['id'], project_id=self.project_baz['id'])
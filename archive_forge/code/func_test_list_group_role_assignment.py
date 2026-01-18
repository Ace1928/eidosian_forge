from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_group_role_assignment(self):
    test_plan = {'entities': {'domains': {'id': CONF.identity.default_domain_id, 'groups': 1, 'projects': 1}, 'roles': 1}, 'assignments': [{'group': 0, 'role': 0, 'project': 0}], 'tests': [{'params': {}, 'results': [{'group': 0, 'role': 0, 'project': 0}]}]}
    self.execute_assignment_plan(test_plan)
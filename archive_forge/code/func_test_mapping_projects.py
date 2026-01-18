import flask
import uuid
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.auth.plugins import mapped
import keystone.conf
from keystone import exception
from keystone.federation import utils as mapping_utils
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from unittest import mock
def test_mapping_projects(self):
    mapping = mapping_fixtures.MAPPING_PROJECTS
    assertion = mapping_fixtures.EMPLOYEE_ASSERTION
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    values = rp.process(assertion)
    self.assertValidMappedUserObject(values)
    expected_username = mapping_fixtures.EMPLOYEE_ASSERTION['UserName']
    self.assertEqual(expected_username, values['user']['name'])
    expected_projects = [{'name': 'Production', 'roles': [{'name': 'observer'}]}, {'name': 'Staging', 'roles': [{'name': 'member'}]}, {'name': 'Project for %s' % expected_username, 'roles': [{'name': 'admin'}]}]
    self.assertEqual(expected_projects, values['projects'])
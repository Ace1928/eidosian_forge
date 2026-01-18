import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
from keystone.tests.unit import utils as test_utils
def test_get_project_with_include_limits(self):
    PROVIDERS.assignment_api.create_system_grant_for_user(self.user_id, self.role_id)
    system_admin_token = self.get_system_scoped_token()
    parent, project, subproject = self._create_projects_hierarchy(2)
    for proj in (parent, project, subproject):
        self.put(self.build_role_assignment_link(role_id=self.role_id, user_id=self.user_id, project_id=proj['project']['id']))
    reg_limit = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id, resource_name='volume')
    self.post('/registered_limits', body={'registered_limits': [reg_limit]}, token=system_admin_token, expected_status=http.client.CREATED)
    limit1 = unit.new_limit_ref(project_id=parent['project']['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume')
    limit2 = unit.new_limit_ref(project_id=project['project']['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume')
    limit3 = unit.new_limit_ref(project_id=subproject['project']['id'], service_id=self.service_id, region_id=self.region_id, resource_name='volume')
    self.post('/limits', body={'limits': [limit1, limit2, limit3]}, token=system_admin_token, expected_status=http.client.CREATED)
    r = self.get('/projects/%(project_id)s?include_limits' % {'project_id': subproject['project']['id']})
    self.assertNotIn('parents', r.result['project'])
    self.assertNotIn('subtree', r.result['project'])
    self.assertNotIn('limits', r.result['project'])
    r = self.get('/projects/%(project_id)s?include_limits&parents_as_list' % {'project_id': subproject['project']['id']})
    self.assertEqual(2, len(r.result['project']['parents']))
    for parent in r.result['project']['parents']:
        self.assertEqual(1, len(parent['project']['limits']))
        self.assertEqual(parent['project']['id'], parent['project']['limits'][0]['project_id'])
        self.assertEqual(10, parent['project']['limits'][0]['resource_limit'])
    r = self.get('/projects/%(project_id)s?include_limits&subtree_as_list' % {'project_id': parent['project']['id']})
    self.assertEqual(2, len(r.result['project']['subtree']))
    for child in r.result['project']['subtree']:
        self.assertEqual(1, len(child['project']['limits']))
        self.assertEqual(child['project']['id'], child['project']['limits'][0]['project_id'])
        self.assertEqual(10, child['project']['limits'][0]['resource_limit'])
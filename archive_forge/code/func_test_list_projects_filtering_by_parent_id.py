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
def test_list_projects_filtering_by_parent_id(self):
    """Call ``GET /projects?parent_id={project_id}``."""
    projects = self._create_projects_hierarchy(hierarchy_size=2)
    new_ref = unit.new_project_ref(domain_id=self.domain_id, parent_id=projects[1]['project']['id'])
    resp = self.post('/projects', body={'project': new_ref})
    self.assertValidProjectResponse(resp, new_ref)
    projects.append(resp.result)
    r = self.get('/projects?parent_id=%(project_id)s' % {'project_id': projects[0]['project']['id']})
    self.assertValidProjectListResponse(r)
    projects_result = r.result['projects']
    expected_list = [projects[1]['project']]
    self.assertEqual(expected_list, projects_result)
    r = self.get('/projects?parent_id=%(project_id)s' % {'project_id': projects[1]['project']['id']})
    self.assertValidProjectListResponse(r)
    projects_result = r.result['projects']
    expected_list = [projects[2]['project'], projects[3]['project']]
    self.assertEqual(expected_list, projects_result)
    r = self.get('/projects?parent_id=%(project_id)s' % {'project_id': projects[2]['project']['id']})
    self.assertValidProjectListResponse(r)
    projects_result = r.result['projects']
    expected_list = []
    self.assertEqual(expected_list, projects_result)
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
def test_list_projects_by_tags_and_name(self):
    """Call ``GET /projects?tags-any={tags}&name={name}``."""
    project, tags = self._create_project_and_tags(num_of_tags=2)
    ref = {'project': {'name': 'tags and name'}}
    resp = self.patch('/projects/%(project_id)s' % {'project_id': project['id']}, body=ref)
    url = '/projects?tags-any=%(values)s&name=%(name)s'
    resp = self.get(url % {'values': tags[0], 'name': 'tags and name'})
    self.assertValidProjectListResponse(resp)
    pids = [p['id'] for p in resp.result['projects']]
    self.assertIn(project['id'], pids)
    resp = self.get(url % {'values': tags[0], 'name': 'foo'})
    self.assertValidProjectListResponse(resp)
    self.assertEqual(len(resp.result['projects']), 0)
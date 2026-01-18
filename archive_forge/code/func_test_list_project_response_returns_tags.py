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
def test_list_project_response_returns_tags(self):
    """Call ``GET /projects`` should always return tag attributes."""
    tagged_project, tags = self._create_project_and_tags()
    self.get('/projects')
    ref = unit.new_project_ref(domain_id=self.domain_id)
    untagged_project = self.post('/projects', body={'project': ref}).json_body['project']
    resp = self.get('/projects')
    for project in resp.json_body['projects']:
        if project['id'] == tagged_project['id']:
            self.assertIsNotNone(project['tags'])
            self.assertEqual(project['tags'], tags)
        if project['id'] == untagged_project['id']:
            self.assertEqual(project['tags'], [])
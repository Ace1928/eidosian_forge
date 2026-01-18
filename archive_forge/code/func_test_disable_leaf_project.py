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
def test_disable_leaf_project(self):
    """Call ``PATCH /projects/{project_id}``."""
    projects = self._create_projects_hierarchy()
    leaf_project = projects[1]['project']
    leaf_project['enabled'] = False
    r = self.patch('/projects/%(project_id)s' % {'project_id': leaf_project['id']}, body={'project': leaf_project})
    self.assertEqual(leaf_project['enabled'], r.result['project']['enabled'])
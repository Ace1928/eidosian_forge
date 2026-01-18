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
def test_disable_not_leaf_project(self):
    """Call ``PATCH /projects/{project_id}``."""
    projects = self._create_projects_hierarchy()
    root_project = projects[0]['project']
    root_project['enabled'] = False
    self.patch('/projects/%(project_id)s' % {'project_id': root_project['id']}, body={'project': root_project}, expected_status=http.client.FORBIDDEN)
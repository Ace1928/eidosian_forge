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
def test_list_head_projects(self):
    """Call ``GET & HEAD /projects``."""
    resource_url = '/projects'
    r = self.get(resource_url)
    self.assertValidProjectListResponse(r, ref=self.project, resource_url=resource_url)
    self.head(resource_url, expected_status=http.client.OK)
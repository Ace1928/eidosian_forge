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
def test_get_project_tag_that_does_not_exist(self):
    project, _ = self._create_project_and_tags()
    self.get('/projects/%(project_id)s/tags/%(value)s' % {'project_id': project['id'], 'value': uuid.uuid4().hex}, expected_status=http.client.NOT_FOUND)
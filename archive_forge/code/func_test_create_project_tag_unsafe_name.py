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
def test_create_project_tag_unsafe_name(self):
    tag = uuid.uuid4().hex + ','
    self.put('/projects/%(project_id)s/tags/%(value)s' % {'project_id': self.project_id, 'value': tag}, expected_status=http.client.BAD_REQUEST)
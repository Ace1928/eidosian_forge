import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import grant as gp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_can_create_grant_for_user_on_project(self):
    user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
    project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=self.domain_id))
    with self.test_client() as c:
        c.put('/v3/projects/%s/users/%s/roles/%s' % (project['id'], user['id'], self.bootstrapper.reader_role_id), headers=self.headers)
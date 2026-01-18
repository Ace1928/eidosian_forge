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
def test_can_check_grant_for_user_on_domain(self):
    user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
    PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=user['id'], domain_id=self.domain_id)
    with self.test_client() as c:
        c.get('/v3/domains/%s/users/%s/roles/%s' % (self.domain_id, user['id'], self.bootstrapper.reader_role_id), headers=self.headers, expected_status_code=http.client.NO_CONTENT)
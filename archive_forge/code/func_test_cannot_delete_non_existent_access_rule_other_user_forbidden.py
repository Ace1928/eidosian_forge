import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_cannot_delete_non_existent_access_rule_other_user_forbidden(self):
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user = PROVIDERS.identity_api.create_user(user)
    with self.test_client() as c:
        c.delete('/v3/users/%s/access_rules/%s' % (user['id'], uuid.uuid4().hex), headers=self.headers, expected_status_code=http.client.FORBIDDEN)
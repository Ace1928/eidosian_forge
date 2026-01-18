import datetime
from testtools import matchers
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_head_application_credential_not_found(self):
    with self.test_client() as c:
        token = self.get_scoped_token()
        c.head('/v3%s' % MEMBER_PATH_FMT % {'user_id': self.user_id, 'app_cred_id': uuid.uuid4().hex}, expected_status_code=http.client.NOT_FOUND, headers={'X-Auth-Token': token})
        c.get('/v3%s' % MEMBER_PATH_FMT % {'user_id': self.user_id, 'app_cred_id': uuid.uuid4().hex}, expected_status_code=http.client.NOT_FOUND, headers={'X-Auth-Token': token})
import datetime
from testtools import matchers
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_application_credentials_with_deleted_role(self):
    second_role = unit.new_role_ref(name='test_new_role')
    PROVIDERS.role_api.create_role(second_role['id'], second_role)
    PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_id, self.project_id, second_role['id'])
    with self.test_client() as c:
        token = self.get_scoped_token()
        resp = c.get('/v3/users/%s/application_credentials' % self.user_id, expected_status_code=http.client.OK, headers={'X-Auth-Token': token})
        self.assertEqual([], resp.json['application_credentials'])
        roles = [{'id': second_role['id']}]
        app_cred_body = self._app_cred_body(roles=roles)
        c.post('/v3/users/%s/application_credentials' % self.user_id, json=app_cred_body, expected_status_code=http.client.CREATED, headers={'X-Auth-Token': token})
        resp = c.get('/v3/users/%s/application_credentials' % self.user_id, expected_status_code=http.client.OK, headers={'X-Auth-Token': token})
        PROVIDERS.role_api.delete_role(second_role['id'])
        resp = c.get('/v3/users/%s/application_credentials' % self.user_id, expected_status_code=http.client.OK, headers={'X-Auth-Token': token})
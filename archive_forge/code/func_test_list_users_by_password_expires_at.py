import datetime
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import filtering
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
def test_list_users_by_password_expires_at(self):
    """Ensure users can be filtered on no operator, eq and neq.

        GET /users?password_expires_at={timestamp}
        GET /users?password_expires_at=eq:{timestamp}

        """
    expire_at_url = self._list_users_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2)))
    resp_users = self.get(expire_at_url).result.get('users')
    self.assertEqual(self.user2['id'], resp_users[0]['id'])
    expire_at_url = self._list_users_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2)), 'eq')
    resp_users = self.get(expire_at_url).result.get('users')
    self.assertEqual(self.user2['id'], resp_users[0]['id'])
    expire_at_url = self._list_users_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2)), 'neq')
    resp_users = self.get(expire_at_url).result.get('users')
    self.assertEqual(self.user['id'], resp_users[0]['id'])
    self.assertEqual(self.user3['id'], resp_users[1]['id'])
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
def test_list_users_in_group_by_password_expires_before(self):
    """Ensure users in a group can be filtered on with lt and lte.

        GET /groups/{groupid}/users?password_expires_at=lt:{timestamp}
        GET /groups/{groupid}/users?password_expires_at=lte:{timestamp}

        """
    expire_before_url = self._list_users_in_group_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2, seconds=1)), 'lt')
    resp_users = self.get(expire_before_url).result.get('users')
    self.assertEqual(self.user2['id'], resp_users[0]['id'])
    expire_before_url = self._list_users_in_group_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2)), 'lte')
    resp_users = self.get(expire_before_url).result.get('users')
    self.assertEqual(self.user2['id'], resp_users[0]['id'])
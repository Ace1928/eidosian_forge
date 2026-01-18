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
def test_list_users_filtered_by_funny_name(self):
    """GET /users?name=%myname%.

        Test Plan:

        - Update policy so api is unprotected
        - Update a user with name that has filter escape characters
        - Ensure we can filter on it

        """
    time = datetime.datetime.utcnow()
    with freezegun.freeze_time(time) as frozen_datetime:
        self._set_policy({'identity:list_users': []})
        user = self.user1
        user['name'] = '%my%name%'
        PROVIDERS.identity_api.update_user(user['id'], user)
        frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
        url_by_name = '/users?name=%my%name%'
        r = self.get(url_by_name, auth=self.auth)
        self.assertEqual(1, len(r.result.get('users')))
        self.assertEqual(user['id'], r.result.get('users')[0]['id'])
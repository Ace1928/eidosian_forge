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
def test_list_users_filtered_by_domain(self):
    """GET /users?domain_id=mydomain (filtered).

        Test Plan:

        - Update policy so api is unprotected
        - Use an un-scoped token to make sure we can filter the
          users by domainB, getting back the 2 users in that domain

        """
    self._set_policy({'identity:list_users': []})
    url_by_name = '/users?domain_id=%s' % self.domainB['id']
    r = self.get(url_by_name, auth=self.auth)
    id_list = self._get_id_list_from_ref_list(r.result.get('users'))
    self.assertIn(self.user2['id'], id_list)
    self.assertIn(self.user3['id'], id_list)
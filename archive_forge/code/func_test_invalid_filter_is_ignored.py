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
def test_invalid_filter_is_ignored(self):
    """GET /domains?enableds&name=myname.

        Test Plan:

        - Update policy for no protection on api
        - Filter by name and 'enableds', which does not exist
        - Assert 'enableds' is ignored

        """
    new_policy = {'identity:list_domains': []}
    self._set_policy(new_policy)
    my_url = '/domains?enableds=0&name=%s' % self.domainA['name']
    r = self.get(my_url, auth=self.auth)
    id_list = self._get_id_list_from_ref_list(r.result.get('domains'))
    self.assertEqual(1, len(id_list))
    self.assertIn(self.domainA['id'], id_list)
    self.assertIs(True, r.result.get('domains')[0]['enabled'])
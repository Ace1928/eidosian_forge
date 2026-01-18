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
def test_at_limit(self):
    """Check truncated attribute not set when list at max size."""
    self._set_policy({'identity:list_services': []})
    self.config_fixture.config(list_limit=5)
    self.config_fixture.config(group='catalog', list_limit=10)
    r = self.get('/services', auth=self.auth)
    self.assertEqual(10, len(r.result.get('services')))
    self.assertNotIn('truncated', r.result)
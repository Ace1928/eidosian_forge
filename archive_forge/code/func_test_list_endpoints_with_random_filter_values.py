import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_list_endpoints_with_random_filter_values(self):
    """Call ``GET /endpoints?interface={interface}...``.

        Ensure passing random values for: interface, region_id and
        service_id will return an empty list.

        """
    self._create_random_endpoint(interface='internal')
    response = self.get('/endpoints?interface=%s' % uuid.uuid4().hex)
    self.assertEqual(0, len(response.json['endpoints']))
    response = self.get('/endpoints?region_id=%s' % uuid.uuid4().hex)
    self.assertEqual(0, len(response.json['endpoints']))
    response = self.get('/endpoints?service_id=%s' % uuid.uuid4().hex)
    self.assertEqual(0, len(response.json['endpoints']))
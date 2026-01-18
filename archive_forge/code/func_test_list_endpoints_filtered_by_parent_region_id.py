import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_list_endpoints_filtered_by_parent_region_id(self):
    """Call ``GET /endpoints?region_id={region_id}``.

        Ensure passing the parent_region_id as filter returns an
        empty list.

        """
    parent_region = self._create_region_with_parent_id()
    parent_region_id = parent_region.result['region']['id']
    self._create_random_endpoint(parent_region_id=parent_region_id)
    response = self.get('/endpoints?region_id=%s' % parent_region_id)
    self.assertEqual(0, len(response.json['endpoints']))
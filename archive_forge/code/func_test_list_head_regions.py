import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_list_head_regions(self):
    """Call ``GET & HEAD /regions``."""
    resource_url = '/regions'
    r = self.get(resource_url)
    self.assertValidRegionListResponse(r, ref=self.region)
    self.head(resource_url, expected_status=http.client.OK)
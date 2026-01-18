import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_create_regions_without_descriptions(self):
    """Call ``POST /regions`` with no description."""
    ref1 = unit.new_region_ref()
    ref2 = unit.new_region_ref()
    del ref1['description']
    ref2['description'] = None
    resp1 = self.post('/regions', body={'region': ref1})
    resp2 = self.post('/regions', body={'region': ref2})
    ref1['description'] = ''
    ref2['description'] = ''
    self.assertValidRegionResponse(resp1, ref1)
    self.assertValidRegionResponse(resp2, ref2)
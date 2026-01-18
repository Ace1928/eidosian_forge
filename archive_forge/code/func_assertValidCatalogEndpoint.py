import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def assertValidCatalogEndpoint(self, entity, ref=None):
    keys = ['description', 'id', 'interface', 'name', 'region_id', 'url']
    for k in keys:
        self.assertEqual(ref.get(k), entity[k], k)
    self.assertEqual(entity['region_id'], entity['region'])
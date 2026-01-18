from unittest import mock
import uuid
from keystone.catalog.backends import base as catalog_base
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit.catalog import test_backends as catalog_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
@unit.skip_if_cache_is_enabled('catalog')
def test_catalog_ignored_malformed_urls(self):
    catalog_ref = PROVIDERS.catalog_api.get_catalog('foo', 'bar')
    self.assertEqual(2, len(catalog_ref['RegionOne']))
    region = PROVIDERS.catalog_api.driver.templates['RegionOne']
    region['compute']['adminURL'] = 'http://localhost:8774/v1.1/$(tenant)s'
    catalog_ref = PROVIDERS.catalog_api.get_catalog('foo', 'bar')
    self.assertEqual(1, len(catalog_ref['RegionOne']))
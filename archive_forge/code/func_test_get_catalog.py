from unittest import mock
import uuid
from keystone.catalog.backends import base as catalog_base
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit.catalog import test_backends as catalog_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
def test_get_catalog(self):
    catalog_ref = PROVIDERS.catalog_api.get_catalog('foo', 'bar')
    self.assertDictEqual(self.DEFAULT_FIXTURE, catalog_ref)
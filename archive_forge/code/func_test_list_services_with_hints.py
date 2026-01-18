from unittest import mock
import uuid
from keystone.catalog.backends import base as catalog_base
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit.catalog import test_backends as catalog_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
def test_list_services_with_hints(self):
    hints = {}
    services = PROVIDERS.catalog_api.list_services(hints=hints)
    exp_services = [{'type': 'compute', 'description': '', 'enabled': True, 'name': "'Compute Service'", 'id': 'compute'}, {'type': 'identity', 'description': '', 'enabled': True, 'name': "'Identity Service'", 'id': 'identity'}]
    self.assertCountEqual(exp_services, services)
import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
@unit.skip_if_cache_disabled('catalog')
def test_cache_layer_service_crud(self):
    new_service = unit.new_service_ref()
    service_id = new_service['id']
    res = PROVIDERS.catalog_api.create_service(service_id, new_service)
    self.assertDictEqual(new_service, res)
    PROVIDERS.catalog_api.get_service(service_id)
    updated_service = copy.deepcopy(new_service)
    updated_service['description'] = uuid.uuid4().hex
    PROVIDERS.catalog_api.driver.update_service(service_id, updated_service)
    self.assertLessEqual(new_service.items(), PROVIDERS.catalog_api.get_service(service_id).items())
    PROVIDERS.catalog_api.get_service.invalidate(PROVIDERS.catalog_api, service_id)
    self.assertLessEqual(updated_service.items(), PROVIDERS.catalog_api.get_service(service_id).items())
    PROVIDERS.catalog_api.driver.delete_service(service_id)
    self.assertLessEqual(updated_service.items(), PROVIDERS.catalog_api.get_service(service_id).items())
    PROVIDERS.catalog_api.get_service.invalidate(PROVIDERS.catalog_api, service_id)
    self.assertRaises(exception.ServiceNotFound, PROVIDERS.catalog_api.delete_service, service_id)
    self.assertRaises(exception.ServiceNotFound, PROVIDERS.catalog_api.get_service, service_id)
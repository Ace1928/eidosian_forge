import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_cache_layer_delete_service_with_endpoint(self):
    service = unit.new_service_ref()
    PROVIDERS.catalog_api.create_service(service['id'], service)
    endpoint = unit.new_endpoint_ref(service_id=service['id'], region_id=None)
    PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
    PROVIDERS.catalog_api.get_service(service['id'])
    PROVIDERS.catalog_api.get_endpoint(endpoint['id'])
    PROVIDERS.catalog_api.driver.delete_service(service['id'])
    self.assertLessEqual(endpoint.items(), PROVIDERS.catalog_api.get_endpoint(endpoint['id']).items())
    self.assertLessEqual(service.items(), PROVIDERS.catalog_api.get_service(service['id']).items())
    PROVIDERS.catalog_api.get_endpoint.invalidate(PROVIDERS.catalog_api, endpoint['id'])
    self.assertRaises(exception.EndpointNotFound, PROVIDERS.catalog_api.get_endpoint, endpoint['id'])
    self.assertRaises(exception.EndpointNotFound, PROVIDERS.catalog_api.delete_endpoint, endpoint['id'])
    second_endpoint = unit.new_endpoint_ref(service_id=service['id'], region_id=None)
    PROVIDERS.catalog_api.create_service(service['id'], service)
    PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
    PROVIDERS.catalog_api.create_endpoint(second_endpoint['id'], second_endpoint)
    PROVIDERS.catalog_api.delete_service(service['id'])
    self.assertRaises(exception.EndpointNotFound, PROVIDERS.catalog_api.get_endpoint, endpoint['id'])
    self.assertRaises(exception.EndpointNotFound, PROVIDERS.catalog_api.delete_endpoint, endpoint['id'])
    self.assertRaises(exception.EndpointNotFound, PROVIDERS.catalog_api.get_endpoint, second_endpoint['id'])
    self.assertRaises(exception.EndpointNotFound, PROVIDERS.catalog_api.delete_endpoint, second_endpoint['id'])
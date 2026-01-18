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
def test_invalidate_cache_when_updating_endpoint(self):
    service = unit.new_service_ref()
    PROVIDERS.catalog_api.create_service(service['id'], service)
    endpoint = unit.new_endpoint_ref(service_id=service['id'], region_id=None)
    PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
    PROVIDERS.catalog_api.get_endpoint(endpoint['id'])
    new_url = {'url': uuid.uuid4().hex}
    PROVIDERS.catalog_api.update_endpoint(endpoint['id'], new_url)
    current_endpoint = PROVIDERS.catalog_api.get_endpoint(endpoint['id'])
    self.assertEqual(new_url['url'], current_endpoint['url'])
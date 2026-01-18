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
def test_invalidate_cache_when_updating_service(self):
    new_service = unit.new_service_ref()
    service_id = new_service['id']
    PROVIDERS.catalog_api.create_service(service_id, new_service)
    PROVIDERS.catalog_api.get_service(service_id)
    new_type = {'type': uuid.uuid4().hex}
    PROVIDERS.catalog_api.update_service(service_id, new_type)
    current_service = PROVIDERS.catalog_api.get_service(service_id)
    self.assertEqual(new_type['type'], current_service['type'])
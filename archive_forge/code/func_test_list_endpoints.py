import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_list_endpoints(self):
    service = unit.new_service_ref()
    PROVIDERS.catalog_api.create_service(service['id'], service)
    expected_ids = set([uuid.uuid4().hex for _ in range(3)])
    for endpoint_id in expected_ids:
        endpoint = unit.new_endpoint_ref(service_id=service['id'], id=endpoint_id, region_id=None)
        PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
    endpoints = PROVIDERS.catalog_api.list_endpoints()
    self.assertEqual(expected_ids, set((e['id'] for e in endpoints)))
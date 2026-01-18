import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_create_region_invalid_parent_region_returns_not_found(self):
    new_region = unit.new_region_ref(parent_region_id=uuid.uuid4().hex)
    self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.create_region, new_region)
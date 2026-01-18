import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_list_regions_filtered_by_parent_region_id(self):
    new_region = self._create_region_with_parent_id()
    parent_id = new_region['id']
    new_region = self._create_region_with_parent_id(parent_id)
    new_region = self._create_region_with_parent_id(parent_id)
    hints = driver_hints.Hints()
    hints.add_filter('parent_region_id', parent_id)
    regions = PROVIDERS.catalog_api.list_regions(hints)
    for region in regions:
        self.assertEqual(parent_id, region['parent_region_id'])
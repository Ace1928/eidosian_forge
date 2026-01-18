import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
@mock.patch.object(base.CatalogDriverBase, '_ensure_no_circle_in_hierarchical_regions')
def test_circular_regions_can_be_deleted(self, mock_ensure_on_circle):
    mock_ensure_on_circle.return_value = None
    region_one = self._create_region_with_parent_id()
    PROVIDERS.catalog_api.update_region(region_one['id'], {'parent_region_id': region_one['id']})
    PROVIDERS.catalog_api.delete_region(region_one['id'])
    self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, region_one['id'])
    region_one = self._create_region_with_parent_id()
    region_two = self._create_region_with_parent_id(region_one['id'])
    PROVIDERS.catalog_api.update_region(region_one['id'], {'parent_region_id': region_two['id']})
    PROVIDERS.catalog_api.delete_region(region_one['id'])
    self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, region_one['id'])
    self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, region_two['id'])
    region_one = self._create_region_with_parent_id()
    region_two = self._create_region_with_parent_id(region_one['id'])
    region_three = self._create_region_with_parent_id(region_two['id'])
    PROVIDERS.catalog_api.update_region(region_one['id'], {'parent_region_id': region_three['id']})
    PROVIDERS.catalog_api.delete_region(region_two['id'])
    self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, region_two['id'])
    self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, region_one['id'])
    self.assertRaises(exception.RegionNotFound, PROVIDERS.catalog_api.get_region, region_three['id'])
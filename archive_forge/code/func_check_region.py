import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def check_region(self, region, region_ref=None):
    self.assertIsNotNone(region.id)
    self.assertIn('self', region.links)
    self.assertIn('/regions/' + region.id, region.links['self'])
    if hasattr(region_ref, 'description'):
        self.assertEqual(region_ref['description'], region.description)
    if hasattr(region_ref, 'parent_region'):
        self.assertEqual(region_ref['parent_region'], region.parent_region)
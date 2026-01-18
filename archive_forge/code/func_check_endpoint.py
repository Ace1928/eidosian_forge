import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def check_endpoint(self, endpoint, endpoint_ref=None):
    self.assertIsNotNone(endpoint.id)
    self.assertIn('self', endpoint.links)
    self.assertIn('/endpoints/' + endpoint.id, endpoint.links['self'])
    if endpoint_ref:
        self.assertEqual(endpoint_ref['service'], endpoint.service_id)
        self.assertEqual(endpoint_ref['url'], endpoint.url)
        self.assertEqual(endpoint_ref['interface'], endpoint.interface)
        self.assertEqual(endpoint_ref['enabled'], endpoint.enabled)
        if hasattr(endpoint_ref, 'region'):
            self.assertEqual(endpoint_ref['region'], endpoint.region)
    else:
        self.assertIsNotNone(endpoint.service_id)
        self.assertIsNotNone(endpoint.url)
        self.assertIsNotNone(endpoint.interface)
        self.assertIsNotNone(endpoint.enabled)
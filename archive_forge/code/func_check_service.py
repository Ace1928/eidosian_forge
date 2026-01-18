import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def check_service(self, service, service_ref=None):
    self.assertIsNotNone(service.id)
    self.assertIn('self', service.links)
    self.assertIn('/services/' + service.id, service.links['self'])
    if service_ref:
        self.assertEqual(service_ref['name'], service.name)
        self.assertEqual(service_ref['enabled'], service.enabled)
        self.assertEqual(service_ref['type'], service.type)
        if hasattr(service_ref, 'description'):
            self.assertEqual(service_ref['description'], service.description)
    else:
        self.assertIsNotNone(service.name)
        self.assertIsNotNone(service.enabled)
        self.assertIsNotNone(service.type)
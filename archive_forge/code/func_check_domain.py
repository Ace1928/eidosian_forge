import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def check_domain(self, domain, domain_ref=None):
    self.assertIsNotNone(domain.id)
    self.assertIn('self', domain.links)
    self.assertIn('/domains/' + domain.id, domain.links['self'])
    if domain_ref:
        self.assertEqual(domain_ref['name'], domain.name)
        self.assertEqual(domain_ref['enabled'], domain.enabled)
        if hasattr(domain_ref, 'description'):
            self.assertEqual(domain_ref['description'], domain.description)
    else:
        self.assertIsNotNone(domain.name)
        self.assertIsNotNone(domain.enabled)
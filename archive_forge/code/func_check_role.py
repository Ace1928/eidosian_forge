import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def check_role(self, role, role_ref=None):
    self.assertIsNotNone(role.id)
    self.assertIn('self', role.links)
    self.assertIn('/roles/' + role.id, role.links['self'])
    if role_ref:
        self.assertEqual(role_ref['name'], role.name)
        if hasattr(role_ref, 'domain'):
            self.assertEqual(role_ref['domain'], role.domain_id)
    else:
        self.assertIsNotNone(role.name)
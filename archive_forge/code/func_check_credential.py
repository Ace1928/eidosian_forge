import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def check_credential(self, credential, credential_ref=None):
    self.assertIsNotNone(credential.id)
    self.assertIn('self', credential.links)
    self.assertIn('/credentials/' + credential.id, credential.links['self'])
    if credential_ref:
        self.assertEqual(credential_ref['user'], credential.user_id)
        self.assertEqual(credential_ref['type'], credential.type)
        self.assertEqual(credential_ref['blob'], credential.blob)
        if credential_ref['type'] == 'ec2' or hasattr(credential_ref, 'project'):
            self.assertEqual(credential_ref['project'], credential.project_id)
    else:
        self.assertIsNotNone(credential.user_id)
        self.assertIsNotNone(credential.type)
        self.assertIsNotNone(credential.blob)
        if credential.type == 'ec2':
            self.assertIsNotNone(credential.project_id)
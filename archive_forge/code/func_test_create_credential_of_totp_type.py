import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_create_credential_of_totp_type(self):
    user = fixtures.User(self.client, self.test_domain.id)
    self.useFixture(user)
    credential_ref = {'user': user.id, 'type': 'totp', 'blob': uuid.uuid4().hex}
    credential = self.client.credentials.create(**credential_ref)
    self.addCleanup(self.client.credentials.delete, credential)
    self.check_credential(credential, credential_ref)
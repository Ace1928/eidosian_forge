import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneclient import access
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import tokens
def test_authenticate_use_admin_url(self):
    token_fixture = fixture.V2Token()
    token_fixture.set_scope()
    self.stub_auth(json=token_fixture)
    token_ref = self.client.tokens.authenticate(token=uuid.uuid4().hex)
    self.assertEqual(self.TEST_URL + '/tokens', self.requests_mock.last_request.url)
    self.assertIsInstance(token_ref, tokens.Token)
    self.assertEqual(token_fixture.token_id, token_ref.id)
    self.assertEqual(token_fixture.expires_str, token_ref.expires)
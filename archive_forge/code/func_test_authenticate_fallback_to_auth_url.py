import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneclient import access
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import tokens
def test_authenticate_fallback_to_auth_url(self):
    new_auth_url = 'http://keystone.test:5000/v2.0'
    token_fixture = fixture.V2Token()
    self.stub_auth(base_url=new_auth_url, json=token_fixture)
    with self.deprecations.expect_deprecations_here():
        c = client.Client(username=self.TEST_USER, auth_url=new_auth_url, password=uuid.uuid4().hex)
    self.assertIsNone(c.management_url)
    token_ref = c.tokens.authenticate(token=uuid.uuid4().hex)
    self.assertIsInstance(token_ref, tokens.Token)
    self.assertEqual(token_fixture.token_id, token_ref.id)
    self.assertEqual(token_fixture.expires_str, token_ref.expires)
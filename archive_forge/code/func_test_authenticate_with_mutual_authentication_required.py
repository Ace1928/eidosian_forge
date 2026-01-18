import uuid
from keystoneauth1.extras import kerberos
from keystoneauth1 import fixture as ks_fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit.extras.kerberos import base
def test_authenticate_with_mutual_authentication_required(self):
    self.kerberos_mock.mock_auth_success(url=self.token_url, method='GET')
    scoped_id = uuid.uuid4().hex
    scoped_body = ks_fixture.V3Token()
    scoped_body.set_project_scope()
    self.requests_mock.post('%s/auth/tokens' % self.TEST_V3_URL, json=scoped_body, headers={'X-Subject-Token': scoped_id, 'Content-Type': 'application/json'})
    plugin = kerberos.MappedKerberos(auth_url=self.TEST_V3_URL, protocol=self.protocol, identity_provider=self.identity_provider, project_id=scoped_body.project_id, mutual_auth='required')
    sess = session.Session()
    tok = plugin.get_token(sess)
    proj = plugin.get_project_id(sess)
    self.assertEqual(scoped_id, tok)
    self.assertEqual(scoped_body.project_id, proj)
    self.assertEqual(self.kerberos_mock.called_auth_server, True)
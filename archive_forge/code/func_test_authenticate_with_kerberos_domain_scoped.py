from keystoneauth1.extras import kerberos
from keystoneauth1 import session
from keystoneauth1.tests.unit.extras.kerberos import base
def test_authenticate_with_kerberos_domain_scoped(self):
    token_id, token_body = self.kerberos_mock.mock_auth_success()
    a = kerberos.Kerberos(self.TEST_ROOT_URL + 'v3')
    s = session.Session(a)
    token = a.get_token(s)
    self.assertRequestBody()
    self.assertEqual(self.kerberos_mock.challenge_header, self.requests_mock.last_request.headers['Authorization'])
    self.assertEqual(token_id, a.auth_ref.auth_token)
    self.assertEqual(token_id, token)
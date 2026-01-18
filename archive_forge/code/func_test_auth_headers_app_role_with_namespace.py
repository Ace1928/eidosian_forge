import requests_mock
from castellan.key_manager import vault_key_manager
from castellan.tests.unit.key_manager import test_key_manager
@requests_mock.Mocker()
def test_auth_headers_app_role_with_namespace(self, m):
    self.key_mgr._approle_role_id = 'spam'
    self.key_mgr._approle_secret_id = 'secret'
    self.key_mgr._namespace = 'ham'
    m.post('http://127.0.0.1:8200/v1/auth/approle/login', json={'auth': {'client_token': 'token', 'lease_duration': 3600}})
    expected_headers = {'X-Vault-Token': 'token', 'X-Vault-Namespace': 'ham'}
    self.assertEqual(expected_headers, self.key_mgr._build_auth_headers())
from unittest import mock
from unittest.mock import patch
import uuid
import testtools
from troveclient.v1 import backups
@patch('troveclient.v1.backups.mistral_client')
def test_auth_mistral_client(self, mistral_client):
    with patch.object(self.backups.api.client, 'auth') as auth:
        self.backups._get_mistral_client()
        mistral_client.assert_called_with(auth_url=auth.auth_url, username=auth._username, api_key=auth._password, project_name=auth._project_name)
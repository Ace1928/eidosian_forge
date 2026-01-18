import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import aws
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
@mock.patch.object(aws.Credentials, '__init__', return_value=None)
def test_from_file_required_options_only(self, mock_init, tmpdir):
    info = {'audience': AUDIENCE, 'subject_token_type': SUBJECT_TOKEN_TYPE, 'token_url': TOKEN_URL, 'credential_source': self.CREDENTIAL_SOURCE}
    config_file = tmpdir.join('config.json')
    config_file.write(json.dumps(info))
    credentials = aws.Credentials.from_file(str(config_file))
    assert isinstance(credentials, aws.Credentials)
    mock_init.assert_called_once_with(audience=AUDIENCE, subject_token_type=SUBJECT_TOKEN_TYPE, token_url=TOKEN_URL, token_info_url=None, service_account_impersonation_url=None, service_account_impersonation_options={}, client_id=None, client_secret=None, credential_source=self.CREDENTIAL_SOURCE, quota_project_id=None, workforce_pool_user_project=None)
import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
from google.auth import transport
def test_get_project_id_cloud_resource_manager_success(self):
    token_response = self.SUCCESS_RESPONSE.copy()
    token_headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    token_request_data = {'grant_type': 'urn:ietf:params:oauth:grant-type:token-exchange', 'audience': self.AUDIENCE, 'requested_token_type': 'urn:ietf:params:oauth:token-type:access_token', 'subject_token': 'subject_token_0', 'subject_token_type': self.SUBJECT_TOKEN_TYPE, 'scope': 'https://www.googleapis.com/auth/iam'}
    expire_time = (_helpers.utcnow().replace(microsecond=0) + datetime.timedelta(seconds=3600)).isoformat('T') + 'Z'
    expected_expiry = datetime.datetime.strptime(expire_time, '%Y-%m-%dT%H:%M:%SZ')
    impersonation_response = {'accessToken': 'SA_ACCESS_TOKEN', 'expireTime': expire_time}
    impersonation_headers = {'Content-Type': 'application/json', 'x-goog-user-project': self.QUOTA_PROJECT_ID, 'authorization': 'Bearer {}'.format(token_response['access_token'])}
    impersonation_request_data = {'delegates': None, 'scope': self.SCOPES, 'lifetime': '3600s'}
    request = self.make_mock_request(status=http_client.OK, data=self.SUCCESS_RESPONSE.copy(), impersonation_status=http_client.OK, impersonation_data=impersonation_response, cloud_resource_manager_status=http_client.OK, cloud_resource_manager_data=self.CLOUD_RESOURCE_MANAGER_SUCCESS_RESPONSE)
    credentials = self.make_credentials(service_account_impersonation_url=self.SERVICE_ACCOUNT_IMPERSONATION_URL, scopes=self.SCOPES, quota_project_id=self.QUOTA_PROJECT_ID)
    project_id = credentials.get_project_id(request)
    assert project_id == self.PROJECT_ID
    assert len(request.call_args_list) == 3
    self.assert_token_request_kwargs(request.call_args_list[0][1], token_headers, token_request_data)
    self.assert_impersonation_request_kwargs(request.call_args_list[1][1], impersonation_headers, impersonation_request_data)
    assert credentials.valid
    assert credentials.expiry == expected_expiry
    assert not credentials.expired
    assert credentials.token == impersonation_response['accessToken']
    self.assert_resource_manager_request_kwargs(request.call_args_list[2][1], self.PROJECT_NUMBER, {'x-goog-user-project': self.QUOTA_PROJECT_ID, 'authorization': 'Bearer {}'.format(impersonation_response['accessToken'])})
    project_id = credentials.get_project_id(request)
    assert project_id == self.PROJECT_ID
    assert len(request.call_args_list) == 3
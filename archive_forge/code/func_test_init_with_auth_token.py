import testtools
from unittest import mock
from keystoneauth1.exceptions import catalog
from magnumclient.v1 import client
@mock.patch('magnumclient.common.httpclient.HTTPClient')
def test_init_with_auth_token(self, mock_http_client):
    expected_token = 'expected_password'
    expected_magnum_url = 'expected_magnum_url'
    expected_api_version = 'expected_api_version'
    expected_insecure = False
    expected_timeout = 600
    expected_kwargs = {'expected_key': 'expected_value'}
    client.Client(auth_token=expected_token, magnum_url=expected_magnum_url, api_version=expected_api_version, timeout=expected_timeout, insecure=expected_insecure, **expected_kwargs)
    mock_http_client.assert_called_once_with(expected_magnum_url, token=expected_token, api_version=expected_api_version, timeout=expected_timeout, insecure=expected_insecure, **expected_kwargs)
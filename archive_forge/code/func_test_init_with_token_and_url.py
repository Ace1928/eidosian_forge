from unittest import mock
import testtools
from zunclient.v1 import client
@mock.patch('zunclient.common.httpclient.SessionClient')
@mock.patch('keystoneauth1.token_endpoint.Token')
@mock.patch('keystoneauth1.session.Session')
def test_init_with_token_and_url(self, mock_session, mock_token, http_client):
    mock_auth_plugin = mock.Mock()
    mock_token.return_value = mock_auth_plugin
    session = mock.Mock()
    mock_session.return_value = session
    client.Client(auth_token='mytoken', endpoint_override='http://myurl/')
    mock_session.assert_called_once_with(auth=mock_auth_plugin, cert=None, verify=True)
    http_client.assert_called_once_with(endpoint_override='http://myurl/', interface='public', region_name=None, service_name=None, service_type='container', session=session, api_version=None)
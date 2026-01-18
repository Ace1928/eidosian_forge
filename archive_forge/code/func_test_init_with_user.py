from unittest import mock
import testtools
from zunclient.v1 import client
@mock.patch('zunclient.common.httpclient.SessionClient')
@mock.patch('keystoneauth1.loading.get_plugin_loader')
@mock.patch('keystoneauth1.session.Session')
def test_init_with_user(self, mock_session, mock_loader, http_client):
    mock_plugin = mock.Mock()
    mock_loader.return_value = mock_plugin
    client.Client(username='myuser', auth_url='authurl')
    mock_loader.assert_called_once_with('password')
    mock_plugin.load_from_options.assert_called_once_with(auth_url='authurl', username='myuser', password=None, project_domain_id=None, project_domain_name=None, user_domain_id=None, user_domain_name=None, project_id=None, project_name=None)
    http_client.assert_called_once_with(interface='public', region_name=None, service_name=None, service_type='container', session=mock.ANY, api_version=None)
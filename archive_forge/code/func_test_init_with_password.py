import testtools
from unittest import mock
from keystoneauth1.exceptions import catalog
from magnumclient.v1 import client
@mock.patch('magnumclient.common.httpclient.SessionClient')
@mock.patch('magnumclient.v1.client._load_session')
@mock.patch('magnumclient.v1.client._load_service_type', return_value='container-infra')
def test_init_with_password(self, mock_load_service_type, mock_load_session, mock_http_client):
    self._test_init_with_secret(lambda x: client.Client(password=x), mock_load_service_type, mock_load_session, mock_http_client)
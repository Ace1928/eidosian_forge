from unittest import mock
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient.tests.unit import utils
from ironicclient.v1 import client
def test_client_user_api_version(self, http_client_mock):
    endpoint = 'http://ironic:6385'
    os_ironic_api_version = '1.15'
    client.Client(endpoint, session=self.session, os_ironic_api_version=os_ironic_api_version)
    http_client_mock.assert_called_once_with(endpoint_override=endpoint, session=self.session, os_ironic_api_version=os_ironic_api_version, api_version_select_state='user')
from unittest import mock
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient.tests.unit import utils
from ironicclient.v1 import client
def test_client_user_api_version_latest_with_downgrade(self, http_client_mock):
    endpoint = 'http://ironic:6385'
    os_ironic_api_version = 'latest'
    self.assertRaises(ValueError, client.Client, endpoint, session=self.session, allow_api_version_downgrade=True, os_ironic_api_version=os_ironic_api_version)
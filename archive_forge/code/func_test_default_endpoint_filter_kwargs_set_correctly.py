import io
from requests_mock.contrib import fixture
import testtools
from barbicanclient import barbican as barb
from barbicanclient.barbican import Barbican
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.tests import keystone_client_fixtures
def test_default_endpoint_filter_kwargs_set_correctly(self):
    auth_args = '--no-auth --endpoint http://localhost:9311/ --os-project-id project1'
    argv, remainder = self.parser.parse_known_args(auth_args.split())
    barbican_client = self.barbican.create_client(argv)
    httpclient = barbican_client.secrets._api
    self.assertEqual(client._DEFAULT_SERVICE_INTERFACE, httpclient.interface)
    self.assertEqual(client._DEFAULT_SERVICE_TYPE, httpclient.service_type)
    self.assertEqual(client._DEFAULT_API_VERSION, httpclient.version)
    self.assertIsNone(httpclient.service_name)
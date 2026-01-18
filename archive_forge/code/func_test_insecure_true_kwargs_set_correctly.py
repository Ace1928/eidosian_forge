import io
from requests_mock.contrib import fixture
import testtools
from barbicanclient import barbican as barb
from barbicanclient.barbican import Barbican
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.tests import keystone_client_fixtures
def test_insecure_true_kwargs_set_correctly(self):
    auth_args = '--no-auth --endpoint http://localhost:9311/ --os-project-id project1'
    endpoint_filter_args = '--interface public --service-type custom-type --service-name Burrbican --region-name RegionTwo --barbican-api-version v1'
    args = auth_args + ' ' + endpoint_filter_args
    argv, remainder = self.parser.parse_known_args(args.split())
    argv.insecure = True
    argv.os_identity_api_version = '2.0'
    argv.os_tenant_name = 'my_tenant_name'
    barbican_client = self.barbican.create_client(argv)
    httpclient = barbican_client.secrets._api
    self.assertFalse(httpclient.session.verify)
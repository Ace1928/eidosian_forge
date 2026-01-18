from unittest import mock
import swiftclient.client
import testscenarios
import testtools
from testtools import matchers
import time
from heatclient.common import deployment_utils
from heatclient import exc
from heatclient.v1 import software_configs
def test_create_temp_url(self):
    swift_client = mock.MagicMock()
    swift_client.url = 'http://fake-host.com:8080/v1/AUTH_demo'
    swift_client.head_account = mock.Mock(return_value={'x-account-meta-temp-url-key': '123456'})
    swift_client.post_account = mock.Mock()
    uuid_pattern = '[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89aAbB][a-f0-9]{3}-[a-f0-9]{12}'
    url = deployment_utils.create_temp_url(swift_client, 'bar', 60)
    self.assertFalse(swift_client.post_account.called)
    regexp = 'http://fake-host.com:8080/v1/AUTH_demo/bar-%s/%s\\?temp_url_sig=[0-9a-f]{40,64}&temp_url_expires=[0-9]{10}' % (uuid_pattern, uuid_pattern)
    self.assertThat(url, matchers.MatchesRegex(regexp))
    timeout = int(url.split('=')[-1])
    self.assertTrue(timeout < time.time() + 2 * 365 * 24 * 60 * 60)
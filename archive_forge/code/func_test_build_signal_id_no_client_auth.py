from unittest import mock
import swiftclient.client
import testscenarios
import testtools
from testtools import matchers
import time
from heatclient.common import deployment_utils
from heatclient import exc
from heatclient.v1 import software_configs
def test_build_signal_id_no_client_auth(self):
    hc = mock.MagicMock()
    args = mock.MagicMock()
    args.os_no_client_auth = True
    args.signal_transport = 'TEMP_URL_SIGNAL'
    e = self.assertRaises(exc.CommandError, deployment_utils.build_signal_id, hc, args)
    self.assertEqual('Cannot use --os-no-client-auth, auth required to create a Swift TempURL.', str(e))
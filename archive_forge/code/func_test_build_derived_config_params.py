from unittest import mock
import swiftclient.client
import testscenarios
import testtools
from testtools import matchers
import time
from heatclient.common import deployment_utils
from heatclient import exc
from heatclient.v1 import software_configs
def test_build_derived_config_params(self):
    try:
        self.assertEqual(self.result, deployment_utils.build_derived_config_params(action=self.action, source=self.source, name=self.name, input_values=self.input_values, server_id=self.server_id, signal_transport=self.signal_transport, signal_id=self.signal_id))
    except Exception as e:
        if not self.result_error:
            raise e
        self.assertIsInstance(e, self.result_error)
        self.assertEqual(self.result_error_msg, str(e))
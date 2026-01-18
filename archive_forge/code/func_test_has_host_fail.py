from unittest import mock
from heat.tests import common
from heat.tests import utils
def test_has_host_fail(self):
    self._stub_client()
    self.blazar_client.host.list.return_value = []
    self.assertEqual(False, self.blazar_client_plugin.has_host())
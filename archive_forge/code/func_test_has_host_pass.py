from unittest import mock
from heat.tests import common
from heat.tests import utils
def test_has_host_pass(self):
    self._stub_client()
    self.blazar_client.host.list.return_value = ['hosta']
    self.assertEqual(True, self.blazar_client_plugin.has_host())
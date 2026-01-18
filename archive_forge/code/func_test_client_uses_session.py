from unittest import mock
from heat.common import exception as heat_exception
from heat.engine.clients.os import monasca as client_plugin
from heat.tests import common
from heat.tests import utils
def test_client_uses_session(self):
    context = mock.MagicMock()
    monasca_client = client_plugin.MonascaClientPlugin(context=context)
    self.assertIsNotNone(monasca_client._create())
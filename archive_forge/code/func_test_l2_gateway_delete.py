from unittest import mock
from neutronclient.common import exceptions
from neutronclient.v2_0 import client as neutronclient
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_l2_gateway_delete(self):
    self._create_l2_gateway(self.test_template, self.mock_create_reply)
    self.stack.delete()
    self.mockclient.delete_l2_gateway.assert_called_with('d3590f37-b072-4358-9719-71964d84a31c')
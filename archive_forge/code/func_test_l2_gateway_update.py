from unittest import mock
from neutronclient.common import exceptions
from neutronclient.v2_0 import client as neutronclient
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_l2_gateway_update(self):
    self._create_l2_gateway(self.test_template, self.mock_create_reply)
    self.mockclient.update_l2_gateway.return_value = self.mock_update_reply
    self.mockclient.show_l2_gateway.return_value = self.mock_update_reply
    updated_tmpl = template_format.parse(self.test_template_update)
    updated_stack = utils.parse_stack(updated_tmpl)
    self.stack.update(updated_stack)
    ud_l2gw_resource = self.stack['l2gw']
    self.assertIsNone(ud_l2gw_resource.validate())
    self.assertEqual((ud_l2gw_resource.UPDATE, ud_l2gw_resource.COMPLETE), ud_l2gw_resource.state)
    self.assertEqual('d3590f37-b072-4358-9719-71964d84a31c', ud_l2gw_resource.FnGetRefId())
    self.mockclient.update_l2_gateway.assert_called_once_with('d3590f37-b072-4358-9719-71964d84a31c', self.mock_update_req)
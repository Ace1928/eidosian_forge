from unittest import mock
from neutronclient.common import exceptions
from neutronclient.v2_0 import client as neutronclient
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_l2_gateway_create_invalid_seg(self):
    self.mockclient.create_l2_gateway.side_effect = L2GatewaySegmentationRequired()
    template = template_format.parse(self.test_template_invalid_seg)
    self.stack = utils.parse_stack(template)
    scheduler.TaskRunner(self.stack.create)()
    self.l2gw_resource = self.stack['l2gw']
    self.assertIsNone(self.l2gw_resource.validate())
    self.assertEqual('Resource CREATE failed: L2GatewaySegmentationRequired: resources.l2gw: L2 gateway segmentation id must be consistent for all the interfaces', self.stack.status_reason)
    self.assertEqual((self.l2gw_resource.CREATE, self.l2gw_resource.FAILED), self.l2gw_resource.state)
    self.mockclient.create_l2_gateway.assert_called_once_with(self.mock_create_invalid_seg_req)
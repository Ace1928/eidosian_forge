from unittest import mock
from neutronclient.v2_0 import client as neutronclient
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_l2_gateway_connection_create(self):
    self._create_l2_gateway_connection()
    self.assertIsNone(self.l2gwconn_resource.validate())
    self.assertEqual((self.l2gwconn_resource.CREATE, self.l2gwconn_resource.COMPLETE), self.l2gwconn_resource.state)
    self.assertEqual('e491171c-3458-4d85-b3a3-68a7c4a1cacd', self.l2gwconn_resource.FnGetRefId())
    self.mockclient.create_l2_gateway_connection.assert_called_once_with(self.mock_create_req)
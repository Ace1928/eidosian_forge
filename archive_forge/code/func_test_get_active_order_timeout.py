import calendar
from unittest import mock
from barbicanclient import exceptions as barbican_exceptions
from keystoneauth1 import identity
from keystoneauth1 import service_token
from oslo_context import context
from oslo_utils import timeutils
from oslo_utils import uuidutils
from castellan.common import exception
from castellan.common.objects import symmetric_key as sym_key
from castellan.key_manager import barbican_key_manager
from castellan.tests.unit.key_manager import test_key_manager
def test_get_active_order_timeout(self):
    order_ref_url = 'http://localhost:9311/v1/orders/4fe939b7-72bc-49aa-bd1e-e979589858af'
    number_of_retries = self.key_mgr.conf.barbican.number_of_retries
    pending_order = mock.Mock()
    pending_order.status = 'PENDING'
    pending_order.order_ref = order_ref_url
    self.mock_barbican.orders.get.return_value = pending_order
    self.assertRaises(exception.KeyManagerError, self.key_mgr._get_active_order, self.mock_barbican, order_ref_url)
    self.assertEqual(number_of_retries + 1, self.mock_barbican.orders.get.call_count)
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
def test_barbican_endpoint(self):
    endpoint_data = mock.Mock()
    endpoint_data.url = 'http://localhost:9311'
    auth = mock.Mock(spec=['service_catalog'])
    auth.service_catalog.endpoint_data_for.return_value = endpoint_data
    endpoint = self.key_mgr._get_barbican_endpoint(auth, mock.Mock())
    self.assertEqual(endpoint, 'http://localhost:9311')
    auth.service_catalog.endpoint_data_for.assert_called_once_with(service_type='key-manager', interface='public', region_name=None)
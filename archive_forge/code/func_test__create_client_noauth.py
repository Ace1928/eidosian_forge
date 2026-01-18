from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1 import loading as keystone
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from neutron_lib._i18n import _
from neutron_lib.exceptions import placement as n_exc
from neutron_lib import fixture
from neutron_lib.placement import client as place_client
from neutron_lib.tests import _base as base
@mock.patch('neutron_lib.placement.client.NoAuthClient', autospec=True)
def test__create_client_noauth(self, mock_auth_client):
    self.config.placement.auth_type = 'noauth'
    self.config.placement.auth_section = 'placement/'
    self.placement_api_client = place_client.PlacementAPIClient(self.config)
    self.placement_api_client._create_client()
    mock_auth_client.assert_called_once_with('placement/')
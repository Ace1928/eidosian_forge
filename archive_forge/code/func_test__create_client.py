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
@mock.patch.object(keystone, 'load_auth_from_conf_options')
@mock.patch.object(keystone, 'load_session_from_conf_options')
def test__create_client(self, mock_session_from_conf, mock_auth_from_conf):
    self.config.placement.auth_type = 'password'
    self.placement_api_client = place_client.PlacementAPIClient(self.config)
    self.placement_api_client._create_client()
    mock_auth_from_conf.assert_called_once_with(self.config, 'placement')
    mock_session_from_conf.assert_called_once()
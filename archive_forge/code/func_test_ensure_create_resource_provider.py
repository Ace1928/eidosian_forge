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
def test_ensure_create_resource_provider(self):
    self.placement_fixture.mock_put.side_effect = n_exc.PlacementResourceProviderNotFound(resource_provider=RESOURCE_PROVIDER_UUID)
    self.placement_api_client.ensure_resource_provider(RESOURCE_PROVIDER)
    self.placement_fixture.mock_post.assert_called_once_with('/resource_providers', RESOURCE_PROVIDER)
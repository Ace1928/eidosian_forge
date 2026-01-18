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
def test_list_resource_provider_traits_no_rp(self):
    self.placement_fixture.mock_get.side_effect = ks_exc.NotFound()
    self.assertRaises(n_exc.PlacementResourceProviderNotFound, self.placement_api_client.list_resource_provider_traits, RESOURCE_PROVIDER_UUID)
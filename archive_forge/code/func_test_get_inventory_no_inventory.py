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
def test_get_inventory_no_inventory(self):
    _exception = ks_exc.NotFound()
    _exception.details = _('No inventory of class')
    self.placement_fixture.mock_get.side_effect = _exception
    self.assertRaises(n_exc.PlacementInventoryNotFound, self.placement_api_client.get_inventory, RESOURCE_PROVIDER_UUID, RESOURCE_CLASS_NAME)
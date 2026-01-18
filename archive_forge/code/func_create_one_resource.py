import copy
from unittest import mock
from openstack.network.v2 import bgpvpn as _bgpvpn
from openstack import resource as sdk_resource
from osc_lib.utils import columns as column_util
from neutronclient.osc import utils as nc_osc_utils
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.resource_association import\
from neutronclient.osc.v2.networking_bgpvpn.router_association import\
from neutronclient.osc.v2.networking_bgpvpn.router_association import\
from neutronclient.osc.v2.networking_bgpvpn.router_association import\
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
def create_one_resource(attrs=None):
    """Create a fake resource."""
    attrs = attrs or {}
    res_attrs = {'id': 'fake_resource_id', 'tenant_id': _FAKE_PROJECT_ID}
    res_attrs.update(attrs)
    return FakeResource(**res_attrs)
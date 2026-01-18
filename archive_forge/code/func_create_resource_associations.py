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
def create_resource_associations(resources):
    """Create multiple fake resource associations."""
    res_assocs = []
    for idx, resource in enumerate(resources):
        res_assoc_attrs = {'id': 'fake_association_id%d' % idx, 'tenant_id': resource['tenant_id'], 'fake_resource_id': resource['id']}
        res_assocs.append(copy.deepcopy(res_assoc_attrs))
    return res_assocs
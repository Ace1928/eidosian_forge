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
def create_bgpvpns(attrs=None, count=1):
    """Create multiple fake BGP VPN."""
    bgpvpns = []
    for i in range(0, count):
        if attrs is None:
            attrs = {'id': 'fake_id%d' % i}
        elif getattr(attrs, 'id', None) is None:
            attrs['id'] = 'fake_id%d' % i
        bgpvpns.append(create_one_bgpvpn(attrs))
    return bgpvpns
import collections
import contextlib
import hashlib
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from webob import exc as web_exc
from neutron_lib._i18n import _
from neutron_lib.api import attributes
from neutron_lib.api.definitions import network as net_apidef
from neutron_lib.api.definitions import port as port_apidef
from neutron_lib.api.definitions import portbindings as pb
from neutron_lib.api.definitions import portbindings_extended as pb_ext
from neutron_lib.api.definitions import subnet as subnet_apidef
from neutron_lib import constants
from neutron_lib import exceptions
def parse_network_vlan_range(network_vlan_range):
    """Parse a well formed network VLAN range string.

    The network VLAN range string has the format:
        network[:vlan_begin:vlan_end]

    :param network_vlan_range: The network VLAN range string to parse.
    :returns: A tuple who's 1st element is the network name and 2nd
        element is the VLAN range parsed from network_vlan_range.
    :raises: NetworkVlanRangeError if network_vlan_range is malformed.
        PhysicalNetworkNameError if network_vlan_range is missing a network
        name.
    """
    entry = network_vlan_range.strip()
    if ':' in entry:
        if entry.count(':') != 2:
            raise exceptions.NetworkVlanRangeError(vlan_range=entry, error=_('Need exactly two values for VLAN range'))
        network, vlan_min, vlan_max = entry.split(':')
        if not network:
            raise exceptions.PhysicalNetworkNameError()
        try:
            vlan_min = int(vlan_min)
        except ValueError:
            _raise_invalid_tag(vlan_min, entry)
        try:
            vlan_max = int(vlan_max)
        except ValueError:
            _raise_invalid_tag(vlan_max, entry)
        vlan_range = (vlan_min, vlan_max)
        verify_vlan_range(vlan_range)
        return (network, vlan_range)
    else:
        return (entry, None)
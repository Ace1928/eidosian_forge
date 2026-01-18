import logging
from os_ken.services.protocols.bgp.api.base import register
from os_ken.services.protocols.bgp.api.base import RegisterWithArgChecks
from os_ken.services.protocols.bgp.api.base import FLOWSPEC_FAMILY
from os_ken.services.protocols.bgp.api.base import FLOWSPEC_RULES
from os_ken.services.protocols.bgp.api.base import FLOWSPEC_ACTIONS
from os_ken.services.protocols.bgp.core_manager import CORE_MANAGER
from os_ken.services.protocols.bgp.rtconf.base import ConfWithId
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
from os_ken.services.protocols.bgp.rtconf import neighbors
from os_ken.services.protocols.bgp.rtconf.neighbors import NeighborConf
from os_ken.services.protocols.bgp.rtconf.vrfs import ROUTE_DISTINGUISHER
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV4
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfConf
from os_ken.services.protocols.bgp import constants as const
@RegisterWithArgChecks(name='neighbor.in_filter.get', req_args=[neighbors.IP_ADDRESS])
def get_neighbor_in_filter(neigh_ip_address):
    """Returns a neighbor in_filter for given ip address if exists."""
    core = CORE_MANAGER.get_core_service()
    peer = core.peer_manager.get_by_addr(neigh_ip_address)
    return peer.in_filters
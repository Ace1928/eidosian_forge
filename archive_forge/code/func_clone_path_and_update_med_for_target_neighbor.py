import logging
import netaddr
from os_ken.lib import ip
from os_ken.lib.packet.bgp import (
from os_ken.services.protocols.bgp.info_base.rtc import RtcPath
from os_ken.services.protocols.bgp.info_base.ipv4 import Ipv4Path
from os_ken.services.protocols.bgp.info_base.ipv6 import Ipv6Path
from os_ken.services.protocols.bgp.info_base.vpnv4 import Vpnv4Path
from os_ken.services.protocols.bgp.info_base.vpnv6 import Vpnv6Path
from os_ken.services.protocols.bgp.info_base.evpn import EvpnPath
from os_ken.services.protocols.bgp.info_base.ipv4fs import IPv4FlowSpecPath
from os_ken.services.protocols.bgp.info_base.ipv6fs import IPv6FlowSpecPath
from os_ken.services.protocols.bgp.info_base.vpnv4fs import VPNv4FlowSpecPath
from os_ken.services.protocols.bgp.info_base.vpnv6fs import VPNv6FlowSpecPath
from os_ken.services.protocols.bgp.info_base.l2vpnfs import L2VPNFlowSpecPath
def clone_path_and_update_med_for_target_neighbor(path, med):
    assert path and med
    route_family = path.route_family
    if route_family not in _ROUTE_FAMILY_TO_PATH_MAP.keys():
        raise ValueError('Clone is not supported for address-family %s' % route_family)
    path_cls = _ROUTE_FAMILY_TO_PATH_MAP.get(route_family)
    pattrs = path.pathattr_map
    pattrs[BGP_ATTR_TYPE_MULTI_EXIT_DISC] = BGPPathAttributeMultiExitDisc(med)
    return path_cls(path.source, path.nlri, path.source_version_num, pattrs=pattrs, nexthop=path.nexthop, is_withdraw=path.is_withdraw, med_set_by_target_neighbor=True)
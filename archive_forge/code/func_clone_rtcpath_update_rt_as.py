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
def clone_rtcpath_update_rt_as(path, new_rt_as):
    """Clones given RT NLRI `path`, and updates it with new RT_NLRI AS.

        Parameters:
            - `path`: (Path) RT_NLRI path
            - `new_rt_as`: AS value of cloned paths' RT_NLRI
    """
    assert path and new_rt_as
    if not path or path.route_family != RF_RTC_UC:
        raise ValueError('Expected RT_NLRI path')
    old_nlri = path.nlri
    new_rt_nlri = RouteTargetMembershipNLRI(new_rt_as, old_nlri.route_target)
    return RtcPath(path.source, new_rt_nlri, path.source_version_num, pattrs=path.pathattr_map, nexthop=path.nexthop, is_withdraw=path.is_withdraw)
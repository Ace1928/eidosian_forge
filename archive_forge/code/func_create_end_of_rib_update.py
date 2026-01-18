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
def create_end_of_rib_update():
    """Construct end-of-rib (EOR) Update instance."""
    mpunreach_attr = BGPPathAttributeMpUnreachNLRI(RF_IPv4_VPN.afi, RF_IPv4_VPN.safi, [])
    eor = BGPUpdate(path_attributes=[mpunreach_attr])
    return eor
import abc
import logging
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RF_IPv6_UC
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_IPv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_L2VPN_FLOWSPEC
from os_ken.services.protocols.bgp.utils import validation
from os_ken.services.protocols.bgp.base import get_validator
from os_ken.services.protocols.bgp.rtconf.base import BaseConf
from os_ken.services.protocols.bgp.rtconf.base import BaseConfListener
from os_ken.services.protocols.bgp.rtconf.base import ConfigTypeError
from os_ken.services.protocols.bgp.rtconf.base import ConfigValueError
from os_ken.services.protocols.bgp.rtconf.base import ConfWithId
from os_ken.services.protocols.bgp.rtconf.base import ConfWithIdListener
from os_ken.services.protocols.bgp.rtconf.base import ConfWithStats
from os_ken.services.protocols.bgp.rtconf.base import ConfWithStatsListener
from os_ken.services.protocols.bgp.rtconf.base import MAX_NUM_EXPORT_RT
from os_ken.services.protocols.bgp.rtconf.base import MAX_NUM_IMPORT_RT
from os_ken.services.protocols.bgp.rtconf.base import MULTI_EXIT_DISC
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
from os_ken.services.protocols.bgp.rtconf.base import SITE_OF_ORIGINS
from os_ken.services.protocols.bgp.rtconf.base import validate
from os_ken.services.protocols.bgp.rtconf.base import validate_med
from os_ken.services.protocols.bgp.rtconf.base import validate_soo_list
def get_vrf_conf(self, route_dist, vrf_rf, vrf_id=None):
    if route_dist is None and vrf_id is None:
        raise RuntimeConfigError(desc='To get VRF supply route_dist or vrf_id.')
    if route_dist is not None and vrf_id is not None:
        vrf1 = self._vrfs_by_id.get(vrf_id)
        rd_rf_id = VrfConf.create_rd_rf_id(route_dist, vrf_rf)
        vrf2 = self._vrfs_by_rd_rf.get(rd_rf_id)
        if vrf1 is not vrf2:
            raise RuntimeConfigError(desc='Given VRF ID (%s) and RD (%s) are not of same VRF.' % (vrf_id, route_dist))
        vrf = vrf1
    elif route_dist is not None:
        rd_rf_id = VrfConf.create_rd_rf_id(route_dist, vrf_rf)
        vrf = self._vrfs_by_rd_rf.get(rd_rf_id)
    else:
        vrf = self._vrfs_by_id.get(vrf_id)
    return vrf
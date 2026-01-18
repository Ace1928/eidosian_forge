from os_ken.services.protocols.bgp.rtconf.base import ConfWithStats
from os_ken.services.protocols.bgp.rtconf.common import CommonConfListener
from os_ken.services.protocols.bgp.rtconf.neighbors import NeighborsConfListener
from os_ken.services.protocols.bgp.rtconf import vrfs
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfConf
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfsConfListener
import logging
def on_remove_vrf_conf(self, evt):
    """Removes VRF table associated with given `vrf_conf`.

        Cleans up other links to this table as well.
        """
    vrf_conf = evt.value
    vrf_conf.remove_listener(VrfConf.VRF_CHG_EVT, self.on_chg_vrf_conf)
    self._table_manager.remove_vrf_by_vrf_conf(vrf_conf)
    self._rt_manager.update_local_rt_nlris()
    self._signal_bus.vrf_removed(vrf_conf.route_dist)
    rd = vrf_conf.route_dist
    rf = vrf_conf.route_family
    peers = self._peer_manager.iterpeers
    for peer in peers:
        key = ':'.join([rd, rf])
        peer.attribute_maps.pop(key, None)
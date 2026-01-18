from os_ken.services.protocols.bgp.rtconf.base import ConfWithStats
from os_ken.services.protocols.bgp.rtconf.common import CommonConfListener
from os_ken.services.protocols.bgp.rtconf.neighbors import NeighborsConfListener
from os_ken.services.protocols.bgp.rtconf import vrfs
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfConf
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfsConfListener
import logging
def on_add_neighbor_conf(self, evt):
    neigh_conf = evt.value
    self._peer_manager.add_peer(neigh_conf, self._common_config)
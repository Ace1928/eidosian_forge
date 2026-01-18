from os_ken.services.protocols.bgp.rtconf.base import ConfWithStats
from os_ken.services.protocols.bgp.rtconf.common import CommonConfListener
from os_ken.services.protocols.bgp.rtconf.neighbors import NeighborsConfListener
from os_ken.services.protocols.bgp.rtconf import vrfs
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfConf
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfsConfListener
import logging
def on_stats_config_change(self, evt):
    vrf_conf = evt.src
    self._signal_bus.stats_config_changed(vrf_conf)
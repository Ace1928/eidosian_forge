from os_ken.services.protocols.bgp.rtconf.base import ConfWithStats
from os_ken.services.protocols.bgp.rtconf.common import CommonConfListener
from os_ken.services.protocols.bgp.rtconf.neighbors import NeighborsConfListener
from os_ken.services.protocols.bgp.rtconf import vrfs
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfConf
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfsConfListener
import logging
def on_chg_vrf_conf(self, evt):
    evt_value = evt.value
    vrf_conf = evt.src
    new_imp_rts, removed_imp_rts, import_maps, re_export, re_import = evt_value
    route_family = vrf_conf.route_family
    vrf_table = self._table_manager.get_vrf_table(vrf_conf.route_dist, route_family)
    assert vrf_table
    self._table_manager.update_vrf_table_links(vrf_table, new_imp_rts, removed_imp_rts)
    if re_export:
        self._table_manager.re_install_net_ctrl_paths(vrf_table)
    vrf_table.clean_uninteresting_paths()
    if import_maps is not None:
        vrf_table.init_import_maps(import_maps)
        changed_dests = vrf_table.apply_import_maps()
        for dest in changed_dests:
            self._signal_bus.dest_changed(dest)
    if re_import:
        LOG.debug('RE-importing prefixes from VPN table to VRF %r', vrf_table)
        self._table_manager.import_all_vpn_paths_to_vrf(vrf_table)
    else:
        self._table_manager.import_all_vpn_paths_to_vrf(vrf_table, new_imp_rts)
    self._rt_manager.update_local_rt_nlris()
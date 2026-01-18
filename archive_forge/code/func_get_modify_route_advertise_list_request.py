from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_modify_route_advertise_list_request(self, vrf_name, conf_afi, conf_safi, conf_addr_fam):
    request = []
    route_advertise = []
    afi_safi = ('%s_%s' % (conf_afi, conf_safi)).upper()
    route_advertise_list = conf_addr_fam.get('route_advertise_list', [])
    if route_advertise_list:
        for rt_adv in route_advertise_list:
            advertise_afi = rt_adv.get('advertise_afi', None)
            route_map = rt_adv.get('route_map', None)
            if advertise_afi:
                advertise_afi_safi = '%s_UNICAST' % advertise_afi.upper()
                url = '%s=%s/%s' % (self.network_instance_path, vrf_name, self.protocol_bgp_path)
                url += '/%s=%s/%s/route-advertise-list' % (self.afi_safi_path, afi_safi, self.l2vpn_evpn_route_advertise_path)
                cfg = None
                if route_map:
                    route_map_list = [route_map]
                    cfg = {'advertise-afi-safi': advertise_afi_safi, 'route-map': route_map_list}
                else:
                    cfg = {'advertise-afi-safi': advertise_afi_safi}
                route_advertise.append({'advertise-afi-safi': advertise_afi_safi, 'config': cfg})
        pay_load = {'openconfig-bgp-evpn-ext:route-advertise-list': route_advertise}
        request = {'path': url, 'method': PATCH, 'data': pay_load}
    return request
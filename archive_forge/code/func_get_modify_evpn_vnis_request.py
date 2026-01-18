from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_modify_evpn_vnis_request(self, vrf_name, conf_afi, conf_safi, conf_addr_fam):
    request = None
    conf_vnis = conf_addr_fam.get('vnis', [])
    afi_safi = ('%s_%s' % (conf_afi, conf_safi)).upper()
    vnis_dict = {}
    vni_list = []
    if conf_vnis:
        for vni in conf_vnis:
            vni_dict = {}
            cfg = {}
            vni_number = vni.get('vni_number', None)
            adv_default_gw = vni.get('advertise_default_gw', None)
            adv_svi_ip = vni.get('advertise_svi_ip', None)
            rd = vni.get('rd', None)
            rt_in = vni.get('rt_in', [])
            rt_out = vni.get('rt_out', [])
            if vni_number:
                cfg['vni-number'] = vni_number
            if adv_default_gw is not None:
                cfg['advertise-default-gw'] = adv_default_gw
            if adv_svi_ip is not None:
                cfg['advertise-svi-ip'] = adv_svi_ip
            if rd:
                cfg['route-distinguisher'] = rd
            if rt_in:
                cfg['import-rts'] = rt_in
            if rt_out:
                cfg['export-rts'] = rt_out
            if cfg:
                vni_dict['config'] = cfg
                vni_dict['vni-number'] = vni_number
                vni_list.append(vni_dict)
    if vni_list:
        vnis_dict['vni'] = vni_list
        url = '%s=%s/%s/global' % (self.network_instance_path, vrf_name, self.protocol_bgp_path)
        afi_safi_load = {'afi-safi-name': 'openconfig-bgp-types:%s' % afi_safi}
        afi_safi_load['l2vpn-evpn'] = {'openconfig-bgp-evpn-ext:vnis': vnis_dict}
        afi_safis_load = {'afi-safis': {'afi-safi': [afi_safi_load]}}
        pay_load = {'openconfig-network-instance:global': afi_safis_load}
        request = {'path': url, 'method': PATCH, 'data': pay_load}
    return request
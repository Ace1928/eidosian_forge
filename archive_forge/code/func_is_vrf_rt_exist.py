from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_vrf_rt_exist(self):
    """is vpn target exist"""
    if not self.vrf_af_info:
        return False
    for vrf_af_ele in self.vrf_af_info['vpnInstAF']:
        if vrf_af_ele['afType'] == self.vrf_aftype:
            if self.evpn is False:
                if not vrf_af_ele.get('vpnTargets'):
                    return False
                for vpn_target in vrf_af_ele.get('vpnTargets'):
                    if vpn_target['vrfRTType'] == self.vpn_target_type and vpn_target['vrfRTValue'] == self.vpn_target_value:
                        return True
                    else:
                        continue
            else:
                if not vrf_af_ele.get('evpnTargets'):
                    return False
                for evpn_target in vrf_af_ele.get('evpnTargets'):
                    if evpn_target['vrfRTType'] == self.vpn_target_type and evpn_target['vrfRTValue'] == self.vpn_target_value:
                        return True
                    else:
                        continue
        else:
            continue
    return False
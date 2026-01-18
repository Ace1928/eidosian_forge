from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_vrf_rd_exist(self):
    """is vrf route distinguisher exist"""
    if not self.vrf_af_info:
        return False
    for vrf_af_ele in self.vrf_af_info['vpnInstAF']:
        if vrf_af_ele['afType'] == self.vrf_aftype:
            if vrf_af_ele['vrfRD'] is None:
                return False
            if self.route_distinguisher is not None:
                return bool(vrf_af_ele['vrfRD'] == self.route_distinguisher)
            else:
                return True
        else:
            continue
    return False
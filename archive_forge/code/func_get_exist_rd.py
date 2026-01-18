from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_exist_rd(self):
    """get exist route distinguisher """
    if not self.vrf_af_info:
        return None
    for vrf_af_ele in self.vrf_af_info['vpnInstAF']:
        if vrf_af_ele['afType'] == self.vrf_aftype:
            if vrf_af_ele['vrfRD'] is None:
                return None
            else:
                return vrf_af_ele['vrfRD']
        else:
            continue
    return None
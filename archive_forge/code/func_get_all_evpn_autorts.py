from __future__ import (absolute_import, division, print_function)
import re
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_all_evpn_autorts(self, evpn_autorts):
    """"Get all EVPN AUTORTS"""
    autorts = evpn_autorts.findall('evpnAutoRT')
    if not autorts:
        return
    for autort in autorts:
        vrf_rttype = autort.find('vrfRTType')
        if vrf_rttype.text == 'export_extcommunity':
            self.evpn_info['vpn_target_export'].append('auto')
        elif vrf_rttype.text == 'import_extcommunity':
            self.evpn_info['vpn_target_import'].append('auto')
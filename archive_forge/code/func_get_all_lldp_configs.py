from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.lldp_global.lldp_global import Lldp_globalArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_all_lldp_configs(self):
    """Get all the lldp_global configured in the device"""
    request = [{'path': 'data/openconfig-lldp:lldp/config', 'method': GET}]
    lldp_global_data = {}
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    lldp_global_data['tlv_select'] = {}
    lldp_global_data['tlv_select']['management_address'] = True
    lldp_global_data['tlv_select']['system_capabilities'] = True
    lldp_global_data['enable'] = True
    if 'openconfig-lldp:config' in response[0][1]:
        raw_lldp_global_data = response[0][1]['openconfig-lldp:config']
        if 'enabled' in raw_lldp_global_data:
            lldp_global_data['enable'] = raw_lldp_global_data['enabled']
        if 'hello-timer' in raw_lldp_global_data:
            lldp_global_data['hello_time'] = raw_lldp_global_data['hello-timer']
        if 'openconfig-lldp-ext:mode' in raw_lldp_global_data:
            lldp_global_data['mode'] = raw_lldp_global_data['openconfig-lldp-ext:mode'].lower()
        if 'system-description' in raw_lldp_global_data:
            lldp_global_data['system_description'] = raw_lldp_global_data['system-description']
        if 'system-name' in raw_lldp_global_data:
            lldp_global_data['system_name'] = raw_lldp_global_data['system-name']
        if 'openconfig-lldp-ext:multiplier' in raw_lldp_global_data:
            lldp_global_data['multiplier'] = raw_lldp_global_data['openconfig-lldp-ext:multiplier']
        if 'suppress-tlv-advertisement' in raw_lldp_global_data:
            for tlv_select in raw_lldp_global_data['suppress-tlv-advertisement']:
                tlv_select = tlv_select.replace('openconfig-lldp-types:', '').lower()
                if tlv_select in ('management_address', 'system_capabilities'):
                    lldp_global_data['tlv_select'][tlv_select] = False
    return lldp_global_data
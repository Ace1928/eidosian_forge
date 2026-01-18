from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.vlan_mapping.vlan_mapping import Vlan_mappingArgs
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_portchannels(self):
    """Get all portchannel names on device"""
    all_portchannels_path = 'data/sonic-portchannel:sonic-portchannel'
    method = 'GET'
    request = [{'path': all_portchannels_path, 'method': method}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    response = response[0][1]
    portchannel_list = []
    if 'sonic-portchannel:sonic-portchannel' in response:
        component = response['sonic-portchannel:sonic-portchannel']
        if 'PORTCHANNEL' in component:
            component = component['PORTCHANNEL']
            if 'PORTCHANNEL_LIST' in component:
                component = component['PORTCHANNEL_LIST']
                for portchannel in component:
                    portchannel_list.append({'ifname': portchannel['name']})
    return portchannel_list
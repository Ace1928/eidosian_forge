from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.lag_interfaces.lag_interfaces import Lag_interfacesArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_all_portchannels(self):
    """Get all the interfaces available in chassis"""
    request = [{'path': 'data/sonic-portchannel:sonic-portchannel', 'method': GET}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    if response[0][1]:
        data = response[0][1]['sonic-portchannel:sonic-portchannel']
    else:
        data = []
    if data is not None:
        if 'PORTCHANNEL_MEMBER' in data:
            portchannel_members_list = data['PORTCHANNEL_MEMBER']['PORTCHANNEL_MEMBER_LIST']
        else:
            portchannel_members_list = []
        if 'PORTCHANNEL' in data:
            portchannel_list = data['PORTCHANNEL']['PORTCHANNEL_LIST']
        else:
            portchannel_list = []
        if portchannel_list:
            for i in portchannel_list:
                if not any((d['name'] == i['name'] for d in portchannel_members_list)):
                    portchannel_members_list.append({'ifname': None, 'name': i['name']})
    if data:
        return portchannel_members_list
    else:
        return []
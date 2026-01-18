from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.vlan_mapping.vlan_mapping import Vlan_mappingArgs
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_port_mappings(self, interface):
    """Get a ports vlan mappings from device"""
    ifname = interface['ifname']
    if '/' in ifname:
        ifname = ifname.replace('/', '%2F')
    port_mappings = 'data/openconfig-interfaces:interfaces/interface=%s/openconfig-interfaces-ext:mapped-vlans' % ifname
    method = 'GET'
    request = [{'path': port_mappings, 'method': method}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    return response[0][1]
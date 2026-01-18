from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.dhcp_snooping.dhcp_snooping import Dhcp_snoopingArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_dhcp_snooping_top_level(self):
    """Get all DHCP snooping configurations available in chassis"""
    dhcp_snooping_path = 'data/openconfig-dhcp-snooping:dhcp-snooping'
    method = 'GET'
    request = [{'path': dhcp_snooping_path, 'method': method}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    config = {}
    if response[0][1].get('openconfig-dhcp-snooping:dhcp-snooping'):
        config = response[0][1].get('openconfig-dhcp-snooping:dhcp-snooping')
    return config
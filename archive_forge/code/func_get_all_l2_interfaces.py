from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.l2_interfaces.l2_interfaces import L2_interfacesArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_all_l2_interfaces(self):
    """Get all the l2_interfaces available in chassis"""
    l2_interfaces = {}
    request = [{'path': 'data/openconfig-interfaces:interfaces', 'method': GET}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    if 'openconfig-interfaces:interfaces' in response[0][1]:
        interfaces = response[0][1].get('openconfig-interfaces:interfaces', {})
        if interfaces.get('interface'):
            interfaces = interfaces['interface']
            l2_interfaces = self.get_l2_interfaces_from_interfaces(interfaces)
        else:
            l2_interfaces = {}
    return l2_interfaces
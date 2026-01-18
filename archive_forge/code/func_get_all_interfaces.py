from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.interfaces.interfaces import InterfacesArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_all_interfaces(self):
    """Get all the interfaces available in chassis"""
    all_interfaces = {}
    request = [{'path': 'data/openconfig-interfaces:interfaces', 'method': GET}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    if 'openconfig-interfaces:interfaces' in response[0][1]:
        all_interfaces = response[0][1].get('openconfig-interfaces:interfaces', {})
    return all_interfaces['interface']
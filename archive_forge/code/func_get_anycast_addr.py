from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.system.system import SystemArgs
def get_anycast_addr(self):
    """Get system anycast address available in chassis"""
    request = [{'path': 'data/sonic-sag:sonic-sag/SAG_GLOBAL/SAG_GLOBAL_LIST/', 'method': GET}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    if 'sonic-sag:SAG_GLOBAL_LIST' in response[0][1]:
        data = response[0][1]['sonic-sag:SAG_GLOBAL_LIST'][0]
    else:
        data = {}
    return data
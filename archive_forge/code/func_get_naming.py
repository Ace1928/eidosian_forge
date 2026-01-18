from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.system.system import SystemArgs
def get_naming(self):
    """Get interface_naming type available in chassis"""
    request = [{'path': 'data/sonic-device-metadata:sonic-device-metadata/DEVICE_METADATA/DEVICE_METADATA_LIST=localhost', 'method': GET}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    if 'sonic-device-metadata:DEVICE_METADATA_LIST' in response[0][1]:
        intf_data = response[0][1]['sonic-device-metadata:DEVICE_METADATA_LIST']
        if 'intf_naming_mode' in intf_data[0]:
            data = intf_data[0]
        else:
            data = {}
    return data
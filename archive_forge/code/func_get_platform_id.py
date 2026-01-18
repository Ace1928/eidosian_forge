from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_platform_id(module):
    info = get_capabilities(module).get('device_info', {})
    return info.get('network_os_platform', '')
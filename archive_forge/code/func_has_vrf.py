from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def has_vrf(module, vrf):
    global _CONFIGURED_VRFS
    if _CONFIGURED_VRFS is not None:
        return vrf in _CONFIGURED_VRFS
    config = get_config(module)
    _CONFIGURED_VRFS = re.findall('vrf definition (\\S+)', config)
    return vrf in _CONFIGURED_VRFS
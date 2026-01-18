from __future__ import absolute_import, division, print_function
import platform
import re
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import (
def platform_facts(self):
    platform_facts = {}
    resp = get_capabilities(self.module)
    device_info = resp['device_info']
    platform_facts['system'] = device_info['network_os']
    for item in ('model', 'image', 'version', 'platform', 'hostname'):
        val = device_info.get('network_os_%s' % item)
        if val:
            platform_facts[item] = val
    platform_facts['api'] = resp['network_api']
    platform_facts['python_version'] = platform.python_version()
    return platform_facts
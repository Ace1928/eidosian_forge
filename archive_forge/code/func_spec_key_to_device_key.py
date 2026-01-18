from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import (
def spec_key_to_device_key(key):
    device_key = key.replace('_', '-')
    if device_key == 'domain-search':
        device_key += ' domain'
    return device_key
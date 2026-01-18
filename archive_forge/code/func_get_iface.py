from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def get_iface(module, blade):
    """Return Filesystem or None"""
    iface = []
    iface.append(module.params['name'])
    try:
        res = blade.network_interfaces.list_network_interfaces(names=iface)
        return res.items[0]
    except Exception:
        return None
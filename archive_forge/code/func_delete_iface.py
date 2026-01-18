from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_iface(module, blade):
    """Delete Network Interface"""
    changed = True
    if not module.check_mode:
        iface = []
        iface.append(module.params['name'])
        try:
            blade.network_interfaces.delete_network_interfaces(names=iface)
        except Exception:
            module.fail_json(msg='Failed to delete network {0}'.format(module.params['name']))
    module.exit_json(changed=changed)
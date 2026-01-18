from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def modify_subnet(module, blade):
    """Modify Subnet settings"""
    changed = False
    subnet = get_subnet(module, blade)
    subnet_new = []
    subnet_new.append(module.params['name'])
    if module.params['prefix']:
        if module.params['prefix'] != subnet.prefix:
            changed = True
            if not module.check_mode:
                try:
                    blade.subnets.update_subnets(names=subnet_new, subnet=Subnet(prefix=module.params['prefix']))
                except Exception:
                    module.fail_json(msg='Failed to change subnet {0} prefix to {1}'.format(module.params['name'], module.params['prefix']))
    if module.params['vlan']:
        if module.params['vlan'] != subnet.vlan:
            changed = True
            if not module.check_mode:
                try:
                    blade.subnets.update_subnets(names=subnet_new, subnet=Subnet(vlan=module.params['vlan']))
                except Exception:
                    module.fail_json(msg='Failed to change subnet {0} VLAN to {1}'.format(module.params['name'], module.params['vlan']))
    if module.params['gateway']:
        if module.params['gateway'] != subnet.gateway:
            changed = True
            if not module.check_mode:
                try:
                    blade.subnets.update_subnets(names=subnet_new, subnet=Subnet(gateway=module.params['gateway']))
                except Exception:
                    module.fail_json(msg='Failed to change subnet {0} gateway to {1}'.format(module.params['name'], module.params['gateway']))
    if module.params['mtu']:
        if module.params['mtu'] != subnet.mtu:
            changed = True
            if not module.check_mode:
                try:
                    blade.subnets.update_subnets(names=subnet_new, subnet=Subnet(mtu=module.params['mtu']))
                    changed = True
                except Exception:
                    module.fail_json(msg='Failed to change subnet {0} MTU to {1}'.format(module.params['name'], module.params['mtu']))
    module.exit_json(changed=changed)
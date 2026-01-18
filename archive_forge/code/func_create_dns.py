from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def create_dns(module, array):
    """Set DNS settings"""
    changed = False
    current_dns = array.get_dns()
    if current_dns['domain'] != module.params['domain'] or sorted(module.params['nameservers']) != sorted(current_dns['nameservers']):
        try:
            changed = True
            if not module.check_mode:
                array.set_dns(domain=module.params['domain'], nameservers=module.params['nameservers'][0:3])
        except Exception:
            module.fail_json(msg='Set DNS settings failed: Check configuration')
    module.exit_json(changed=changed)
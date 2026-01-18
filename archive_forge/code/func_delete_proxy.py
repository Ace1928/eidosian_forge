from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_proxy(module, blade):
    """Delete proxy settings"""
    changed = False
    current_proxy = blade.support.list_support().items[0].proxy
    if current_proxy != '':
        changed = True
        if not module.check_mode:
            try:
                proxy_settings = Support(proxy='')
                blade.support.update_support(support=proxy_settings)
            except Exception:
                module.fail_json(msg='Delete proxy settigs failed')
    module.exit_json(changed=changed)
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_banner(module, blade):
    """Delete MOTD banner text"""
    changed = True
    if not module.check_mode:
        try:
            blade_settings = PureArray(banner='')
            blade.arrays.update_arrays(array_settings=blade_settings)
        except Exception:
            module.fail_json(msg='Failed to delete current MOTD banner text')
    module.exit_json(changed=changed)
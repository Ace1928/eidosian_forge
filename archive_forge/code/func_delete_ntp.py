from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_ntp(module, blade):
    """Delete NTP Servers"""
    changed = True
    if not module.check_mode:
        if blade.arrays.list_arrays().items[0].ntp_servers != []:
            try:
                blade_settings = PureArray(ntp_servers=[])
                blade.arrays.update_arrays(array_settings=blade_settings)
            except Exception:
                module.fail_json(msg='Deletion of NTP servers failed')
    module.exit_json(changed=changed)
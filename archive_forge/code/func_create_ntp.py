from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def create_ntp(module, blade):
    """Set NTP Servers"""
    changed = True
    if not module.check_mode:
        if not module.params['ntp_servers']:
            module.params['ntp_servers'] = ['0.pool.ntp.org']
        try:
            blade_settings = PureArray(ntp_servers=module.params['ntp_servers'][0:4])
            blade.arrays.update_arrays(array_settings=blade_settings)
        except Exception:
            module.fail_json(msg='Update of NTP servers failed')
    module.exit_json(changed=changed)
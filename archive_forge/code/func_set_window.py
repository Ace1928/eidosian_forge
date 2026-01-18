from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def set_window(module, array):
    """Set Maintenace Window"""
    changed = True
    if not 60 <= module.params['timeout'] <= 86400:
        module.fail_json(msg='Maintenance Window Timeout is out of range (60 to 86400)')
    window = flasharray.MaintenanceWindowPost(timeout=module.params['timeout'] * 1000)
    if not module.check_mode:
        state = array.post_maintenance_windows(names=['environment'], maintenance_window=window)
        if state.status_code != 200:
            module.fail_json(msg='Setting maintenance window failed')
    module.exit_json(changed=changed)
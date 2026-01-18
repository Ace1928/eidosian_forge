from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def enable_alert(module, array):
    """Enable Alert Email"""
    changed = True
    if not module.check_mode:
        changed = False
        try:
            array.enable_alert_recipient(module.params['address'])
            changed = True
        except Exception:
            module.fail_json(msg='Failed to enable alert email: {0}'.format(module.params['address']))
    module.exit_json(changed=changed)
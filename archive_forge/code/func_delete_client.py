from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_client(module, blade):
    changed = True
    if not module.check_mode:
        try:
            blade.delete_api_clients(names=[module.params['name']])
        except Exception:
            module.fail_json(msg='Failed to delete API Client {0}'.format(module.params['name']))
    module.exit_json(changed=changed)
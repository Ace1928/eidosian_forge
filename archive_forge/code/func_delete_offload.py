from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def delete_offload(module, array):
    """Delete offload target"""
    changed = True
    if not module.check_mode:
        res = array.delete_offloads(names=[module.params['name']])
        if res.status_code != 200:
            module.fail_json(msg='Failed to delete offload {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)
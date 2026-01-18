from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def recover_pod(module, array):
    """Recover Deleted Pod"""
    changed = True
    if not module.check_mode:
        try:
            array.recover_pod(module.params['name'])
        except Exception:
            module.fail_json(msg='Recovery of pod {0} failed'.format(module.params['name']))
    module.exit_json(changed=changed)
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def recover_vgroup(module, array):
    """Recover Volume Group"""
    changed = True
    if not module.check_mode:
        try:
            array.recover_vgroup(module.params['name'])
        except Exception:
            module.fail_json(msg='Recovery of volume group {0} failed.'.format(module.params['name']))
    module.exit_json(changed=changed)
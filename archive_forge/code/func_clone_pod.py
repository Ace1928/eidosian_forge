from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def clone_pod(module, array):
    """Create Pod Clone"""
    changed = False
    if get_target(module, array) is None:
        if not get_destroyed_target(module, array):
            changed = True
            if not module.check_mode:
                try:
                    array.clone_pod(module.params['name'], module.params['target'])
                except Exception:
                    module.fail_json(msg='Clone pod {0} to pod {1} failed.'.format(module.params['name'], module.params['target']))
        else:
            module.fail_json(msg='Target pod {0} already exists but deleted.'.format(module.params['target']))
    module.exit_json(changed=changed)
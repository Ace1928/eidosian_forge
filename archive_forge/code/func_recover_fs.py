from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def recover_fs(module, array):
    """Recover a deleted file system"""
    changed = True
    if not module.check_mode:
        try:
            file_system = flasharray.FileSystemPatch(destroyed=False)
            array.patch_file_systems(names=[module.params['name']], file_system=file_system)
        except Exception:
            module.fail_json(msg='Failed to recover file system {0}'.format(module.params['name']))
    module.exit_json(changed=changed)
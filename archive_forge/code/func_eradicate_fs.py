from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def eradicate_fs(module, blade):
    """Eradicate Filesystem"""
    changed = True
    if not module.check_mode:
        try:
            blade.file_systems.delete_file_systems(name=module.params['name'])
        except Exception:
            module.fail_json(msg='Failed to eradicate filesystem {0}.'.format(module.params['name']))
    module.exit_json(changed=changed)
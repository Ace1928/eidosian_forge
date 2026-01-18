from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def rename_fs(module, array):
    """Rename a file system"""
    changed = False
    target_name = module.params['rename']
    if '::' in module.params['name']:
        pod_name = module.params['name'].split('::')[0]
        target_name = pod_name + '::' + module.params['rename']
    try:
        target = list(array.get_file_systems(names=[target_name]).items)[0]
    except Exception:
        target = None
    if not target:
        changed = True
        if not module.check_mode:
            try:
                file_system = flasharray.FileSystemPatch(name=target_name)
                array.patch_file_systems(names=[module.params['name']], file_system=file_system)
            except Exception:
                module.fail_json(msg='Failed to rename file system {0}'.format(module.params['name']))
    else:
        module.fail_json(msg='Target file system {0} already exists'.format(module.params['rename']))
    module.exit_json(changed=changed)
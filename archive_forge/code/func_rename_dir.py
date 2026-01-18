from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def rename_dir(module, array):
    """Rename a file system directory"""
    changed = False
    target = array.get_directories(names=[module.params['filesystem'] + ':' + module.params['rename']])
    if target.status_code != 200:
        if not module.check_mode:
            changed = True
            directory = flasharray.DirectoryPatch(name=module.params['filesystem'] + ':' + module.params['rename'])
            res = array.patch_directories(names=[module.params['filesystem'] + ':' + module.params['name']], directory=directory)
            if res.status_code != 200:
                module.fail_json(msg='Failed to delete file system {0}'.format(module.params['name']))
    else:
        module.fail_json(msg='Target file system {0} already exists'.format(module.params['rename']))
    module.exit_json(changed=changed)
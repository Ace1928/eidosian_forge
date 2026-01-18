from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def disable_ds(module, blade):
    """Disable Directory Service"""
    changed = True
    if not module.check_mode:
        try:
            blade.directory_services.update_directory_services(names=[module.params['dstype']], directory_service=DirectoryService(enabled=False))
        except Exception:
            module.fail_json(msg='Disable {0} Directory Service failed'.format(module.params['dstype']))
    module.exit_json(changed=changed)
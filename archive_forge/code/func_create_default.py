from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def create_default(module, array):
    """Create Default Protection"""
    changed = True
    pg_list = []
    if not module.check_mode:
        for pgroup in range(0, len(module.params['name'])):
            if module.params['scope'] == 'array':
                pg_list.append(flasharray.DefaultProtectionReference(name=module.params['name'][pgroup], type='protection_group'))
            else:
                pg_list.append(flasharray.DefaultProtectionReference(name=module.params['pod'] + '::' + module.params['name'][pgroup], type='protection_group'))
        if module.params['scope'] == 'array':
            protection = flasharray.ContainerDefaultProtection(name='', type='', default_protections=pg_list)
            res = array.patch_container_default_protections(names=[''], container_default_protection=protection)
        else:
            protection = flasharray.ContainerDefaultProtection(name=module.params['pod'], type='pod', default_protections=pg_list)
            res = array.patch_container_default_protections(names=[module.params['pod']], container_default_protection=protection)
        if res.status_code != 200:
            module.fail_json(msg='Failed to set default protection. Error: {0}'.format(res.errors[0].message))
    module.exit_json(changed=changed)
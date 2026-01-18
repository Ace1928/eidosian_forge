from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def delete_default(module, array):
    """Delete Default Protection"""
    changed = True
    if not module.check_mode:
        if module.params['scope'] == 'array':
            protection = flasharray.ContainerDefaultProtection(name='', type='', default_protections=[])
            res = array.patch_container_default_protections(names=[''], container_default_protection=protection)
        else:
            protection = flasharray.ContainerDefaultProtection(name=module.params['pod'], type='pod', default_protections=[])
            res = array.patch_container_default_protections(names=[module.params['pod']], container_default_protection=[])
        if res.status_code != 200:
            module.fail_json(msg='Failed to delete default protection. Error: {0}'.format(res.errors[0].message))
    module.exit_json(changed=changed)
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def delete_vgroup(module, array):
    """Delete Volume Group"""
    changed = True
    if not module.check_mode:
        try:
            array.destroy_vgroup(module.params['name'])
        except Exception:
            module.fail_json(msg='Deleting vgroup {0} failed.'.format(module.params['name']))
    if module.params['eradicate']:
        eradicate_vgroup(module, array)
    module.exit_json(changed=changed)
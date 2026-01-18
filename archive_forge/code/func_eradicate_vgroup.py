from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def eradicate_vgroup(module, array):
    """Eradicate Volume Group"""
    changed = True
    if not module.check_mode:
        try:
            array.eradicate_vgroup(module.params['name'])
        except Exception:
            module.fail_json(msg='Eradicating vgroup {0} failed.'.format(module.params['name']))
    module.exit_json(changed=changed)
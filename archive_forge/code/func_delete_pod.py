from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def delete_pod(module, array):
    """Delete Pod"""
    changed = True
    if not module.check_mode:
        try:
            array.destroy_pod(module.params['name'])
            if module.params['eradicate']:
                try:
                    array.eradicate_pod(module.params['name'])
                except Exception:
                    module.fail_json(msg='Eradicate pod {0} failed.'.format(module.params['name']))
        except Exception:
            module.fail_json(msg='Delete pod {0} failed.'.format(module.params['name']))
    module.exit_json(changed=changed)
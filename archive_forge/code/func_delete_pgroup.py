from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def delete_pgroup(module, array):
    """Delete Protection Group"""
    changed = True
    if not module.check_mode:
        if ':' in module.params['name']:
            if '::' not in module.params['name']:
                try:
                    target = ''.join(module.params['target'])
                    array.destroy_pgroup(module.params['name'], on=target)
                except Exception:
                    module.fail_json(msg='Deleting pgroup {0} failed.'.format(module.params['name']))
            else:
                try:
                    array.destroy_pgroup(module.params['name'])
                except Exception:
                    module.fail_json(msg='Deleting pgroup {0} failed.'.format(module.params['name']))
        else:
            try:
                array.destroy_pgroup(module.params['name'])
            except Exception:
                module.fail_json(msg='Deleting pgroup {0} failed.'.format(module.params['name']))
        if module.params['eradicate']:
            eradicate_pgroup(module, array)
    module.exit_json(changed=changed)
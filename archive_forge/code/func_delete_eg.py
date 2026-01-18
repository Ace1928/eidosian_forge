from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.vexata import (
def delete_eg(module, array, eg):
    changed = False
    eg_name = eg['name']
    if module.check_mode:
        module.exit_json(changed=changed)
    try:
        ok = array.delete_eg(eg['id'])
        if ok:
            module.log(msg='Export group {0} deleted.'.format(eg_name))
            changed = True
        else:
            raise Exception
    except Exception:
        module.fail_json(msg='Export group {0} delete failed.'.format(eg_name))
    module.exit_json(changed=changed)
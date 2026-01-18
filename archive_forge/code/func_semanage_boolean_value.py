from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six import binary_type
from ansible.module_utils._text import to_bytes, to_text
def semanage_boolean_value(module, name, state):
    value = 0
    changed = False
    if state:
        value = 1
    try:
        handle = semanage_get_handle(module)
        semanage_begin_transaction(module, handle)
        cur_value = semanage_get_boolean_value(module, handle, name)
        if cur_value != value:
            changed = True
            if not module.check_mode:
                semanage_set_boolean_value(module, handle, name, value)
                semanage_commit(module, handle)
        semanage_destroy_handle(module, handle)
    except Exception as e:
        module.fail_json(msg=u'Failed to manage policy for boolean %s: %s' % (name, to_text(e)))
    return changed
from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six import binary_type
from ansible.module_utils._text import to_bytes, to_text
def semanage_get_handle(module):
    handle = semanage.semanage_handle_create()
    if not handle:
        module.fail_json(msg='Failed to create semanage library handle')
    managed = semanage.semanage_is_managed(handle)
    if managed <= 0:
        semanage.semanage_handle_destroy(handle)
    if managed < 0:
        module.fail_json(msg='Failed to determine whether policy is manage')
    if managed == 0:
        if os.getuid() == 0:
            module.fail_json(msg='Cannot set persistent booleans without managed policy')
        else:
            module.fail_json(msg='Cannot set persistent booleans; please try as root')
    if semanage.semanage_connect(handle) < 0:
        semanage.semanage_handle_destroy(handle)
        module.fail_json(msg='Failed to connect to semanage')
    return handle
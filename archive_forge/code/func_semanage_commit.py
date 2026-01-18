from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six import binary_type
from ansible.module_utils._text import to_bytes, to_text
def semanage_commit(module, handle, load=0):
    semanage.semanage_set_reload(handle, load)
    if semanage.semanage_commit(handle) < 0:
        semanage.semanage_handle_destroy(handle)
        module.fail_json(msg='Failed to commit changes to semanage')
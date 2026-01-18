from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six import binary_type
from ansible.module_utils._text import to_bytes, to_text
def semanage_destroy_handle(module, handle):
    rc = semanage.semanage_disconnect(handle)
    semanage.semanage_handle_destroy(handle)
    if rc < 0:
        module.fail_json(msg='Failed to disconnect from semanage')
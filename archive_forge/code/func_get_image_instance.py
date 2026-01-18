from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
def get_image_instance(module, client, requested_id, requested_name):
    if requested_id:
        return get_image_by_id(module, client, requested_id)
    else:
        return get_image_by_name(module, client, requested_name)
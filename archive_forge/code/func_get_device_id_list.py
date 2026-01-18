from __future__ import absolute_import, division, print_function
import os
import re
import time
import uuid
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def get_device_id_list(module):
    device_ids = module.params.get('device_ids')
    if isinstance(device_ids, str):
        device_ids = listify_string_name_or_id(device_ids)
    device_ids = [di.strip() for di in device_ids]
    for di in device_ids:
        if not is_valid_uuid(di):
            raise Exception("Device ID '%s' does not seem to be valid" % di)
    if len(device_ids) > MAX_DEVICES:
        raise Exception('You specified too many devices, max is %d' % MAX_DEVICES)
    return device_ids
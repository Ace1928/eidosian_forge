from __future__ import absolute_import, division, print_function
import os
import re
import time
import uuid
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def refresh_device_list(module, packet_conn, devices):
    device_ids = [d.id for d in devices]
    new_device_list = get_existing_devices(module, packet_conn)
    return [d for d in new_device_list if d.id in device_ids]
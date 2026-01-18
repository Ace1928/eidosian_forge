from __future__ import absolute_import, division, print_function
import os
import re
import time
import uuid
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def get_existing_devices(module, packet_conn):
    project_id = module.params.get('project_id')
    return packet_conn.list_devices(project_id, params={'per_page': MAX_DEVICES})
from __future__ import absolute_import, division, print_function
import os
import re
import time
import uuid
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def has_public_ip(addr_list, ip_v):
    return any((a['public'] and a['address_family'] == ip_v and a['address'] for a in addr_list))
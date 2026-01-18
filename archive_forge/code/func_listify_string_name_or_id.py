from __future__ import absolute_import, division, print_function
import os
import re
import time
import uuid
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def listify_string_name_or_id(s):
    if ',' in s:
        return s.split(',')
    else:
        return [s]
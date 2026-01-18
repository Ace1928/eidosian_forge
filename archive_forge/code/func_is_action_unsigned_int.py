from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
import re
from tempfile import NamedTemporaryFile
from datetime import datetime
@classmethod
def is_action_unsigned_int(cls, string_num):
    number = 0
    try:
        number = int(string_num)
    except ValueError:
        return False
    if number >= 0:
        return True
    return False
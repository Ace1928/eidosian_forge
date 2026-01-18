from __future__ import absolute_import, division, print_function
import errno
import os
import shutil
import sys
import time
from pwd import getpwnam, getpwuid
from grp import getgrnam, getgrgid
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
def execute_diff_peek(path):
    """Take a guess as to whether a file is a binary file"""
    b_path = to_bytes(path, errors='surrogate_or_strict')
    appears_binary = False
    try:
        with open(b_path, 'rb') as f:
            head = f.read(8192)
    except Exception:
        pass
    else:
        if b'\x00' in head:
            appears_binary = True
    return appears_binary
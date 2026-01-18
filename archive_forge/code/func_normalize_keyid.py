from __future__ import absolute_import, division, print_function
import re
import os.path
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_native
def normalize_keyid(self, keyid):
    """Ensure a keyid doesn't have a leading 0x, has leading or trailing whitespace, and make sure is uppercase"""
    ret = keyid.strip().upper()
    if ret.startswith('0x'):
        return ret[2:]
    elif ret.startswith('0X'):
        return ret[2:]
    else:
        return ret
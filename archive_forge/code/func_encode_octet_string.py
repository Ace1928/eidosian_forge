from __future__ import absolute_import, division, print_function
import base64
import datetime
import sys
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import ModuleFailException
from ansible_collections.community.crypto.plugins.module_utils.acme.io import (
def encode_octet_string(octet_string):
    if len(octet_string) >= 128:
        raise ModuleFailException('Cannot handle octet strings with more than 128 bytes')
    return b'\x04' + chr(len(octet_string)) + octet_string
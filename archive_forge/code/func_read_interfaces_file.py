from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes
def read_interfaces_file(module, filename):
    with open(filename, 'r') as f:
        return read_interfaces_lines(module, f)
from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes
def get_interface_options(iface_lines):
    return [i for i in iface_lines if i['line_type'] == 'option']
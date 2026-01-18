from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes
def optionDict(line, iface, option, value, address_family):
    return {'line': line, 'iface': iface, 'option': option, 'value': value, 'line_type': 'option', 'address_family': address_family}
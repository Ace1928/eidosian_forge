from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
@staticmethod
def list_to_string(lst):
    if lst is None:
        return None
    else:
        return ','.join(lst)
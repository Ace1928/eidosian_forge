from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
@staticmethod
def bool_to_string(boolean):
    if boolean:
        return 'yes'
    else:
        return 'no'
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def build_diff_value(value):
    if not value:
        return '\n'
    elif len(value) == 1:
        return value[0] + '\n'
    else:
        return value
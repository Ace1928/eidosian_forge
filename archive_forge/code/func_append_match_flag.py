from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
def append_match_flag(rule, param, flag, negatable):
    if param == 'match':
        rule.extend([flag])
    elif negatable and param == 'negate':
        rule.extend(['!', flag])
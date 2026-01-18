from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
def append_csv(rule, param, flag):
    if param:
        rule.extend([flag, ','.join(param)])
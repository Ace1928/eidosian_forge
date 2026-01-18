from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def has_checks(self):
    return len(self._checks) > 0
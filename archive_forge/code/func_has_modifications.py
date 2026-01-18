from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
def has_modifications(self):
    return self.value != self._value
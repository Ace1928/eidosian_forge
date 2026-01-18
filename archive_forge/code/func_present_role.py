from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def present_role(self):
    role = self.get_role()
    if role:
        role = self._update_role(role)
    else:
        role = self._create_role(role)
    return role
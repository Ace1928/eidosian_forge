from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def is_valid_vlan_id(self):
    return 0 < self.vlan < 4095
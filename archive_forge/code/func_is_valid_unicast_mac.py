from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def is_valid_unicast_mac(self):
    mac_re = re.match(self.UNICAST_MAC_REGEX, self.mac)
    return mac_re is not None
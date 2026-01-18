from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
@staticmethod
def enforce_ipv4_cidr_notation(ip4_addresses):
    if ip4_addresses is None:
        return None
    return [address if '/' in address else address + '/32' for address in ip4_addresses]
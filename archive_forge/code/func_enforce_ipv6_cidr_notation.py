from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
@staticmethod
def enforce_ipv6_cidr_notation(ip6_addresses):
    if ip6_addresses is None:
        return None
    return [address if '/' in address else address + '/128' for address in ip6_addresses]
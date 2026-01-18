from __future__ import absolute_import, division, print_function
import uuid
import re
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.common.text.converters import to_native
def parse_subnet_cidr(cidr):
    if '/' not in cidr:
        raise Exception('CIDR expression in wrong format, must be address/prefix_len')
    addr, prefixlen = cidr.split('/')
    try:
        prefixlen = int(prefixlen)
    except ValueError:
        raise Exception('Wrong prefix length in CIDR expression {0}'.format(cidr))
    return (addr, prefixlen)
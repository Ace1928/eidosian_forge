from __future__ import absolute_import, division, print_function
import atexit
import time
import re
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.ansible_release import __version__ as ANSIBLE_VERSION
def is_valid_ip6_addr(ip6_addr):
    """Validates given string as IPv6 address.

    Args:
        ip6_addr (str): string to validate as IPv6 address.

    Returns:
        bool: True if string is valid IPv6 address, else False.
    """
    ip6_addr = ip6_addr.lower()
    ip6_addr_split = ip6_addr.split(':')
    if ip6_addr_split[0] == '':
        ip6_addr_split.pop(0)
    if ip6_addr_split[-1] == '':
        ip6_addr_split.pop(-1)
    if len(ip6_addr_split) > 8:
        return False
    if ip6_addr_split.count('') > 1:
        return False
    elif ip6_addr_split.count('') == 1:
        ip6_addr_split.remove('')
    elif len(ip6_addr_split) != 8:
        return False
    ip6_addr_hextet_regex = re.compile('^[0-9a-f]{1,4}$')
    for ip6_addr_hextet in ip6_addr_split:
        if not bool(ip6_addr_hextet_regex.match(ip6_addr_hextet)):
            return False
    return True
from __future__ import absolute_import, division, print_function
import atexit
import time
import re
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.ansible_release import __version__ as ANSIBLE_VERSION
def ip_prefix_to_netmask(ip_prefix, skip_check=False):
    """Converts IPv4 prefix to netmask.

    Args:
        ip_prefix (str): IPv4 prefix to convert.
        skip_check (bool): Skip validation of IPv4 prefix
            (default: False). Use if you are sure IPv4 prefix is valid.

    Returns:
        str: IPv4 netmask equivalent to given IPv4 prefix if
        IPv4 prefix is valid, else an empty string.
    """
    if skip_check:
        ip_prefix_valid = True
    else:
        ip_prefix_valid = is_valid_ip_prefix(ip_prefix)
    if ip_prefix_valid:
        return '.'.join([str(4294967295 << 32 - int(ip_prefix) >> i & 255) for i in [24, 16, 8, 0]])
    else:
        return ''
import logging
import os
import re
import socket
from urllib import parse
import netaddr
from netaddr.core import INET_PTON
import netifaces
from oslo_utils._i18n import _
def is_ipv6_enabled():
    """Check if IPv6 support is enabled on the platform.

    This api will look into the proc entries of the platform to figure
    out the status of IPv6 support on the platform.

    :returns: True if the platform has IPv6 support, False otherwise.

    .. versionadded:: 1.4
    """
    global _IS_IPV6_ENABLED
    if _IS_IPV6_ENABLED is None:
        disabled_ipv6_path = '/proc/sys/net/ipv6/conf/default/disable_ipv6'
        if os.path.exists(disabled_ipv6_path):
            with open(disabled_ipv6_path, 'r') as f:
                disabled = f.read().strip()
            _IS_IPV6_ENABLED = disabled == '0'
        else:
            _IS_IPV6_ENABLED = False
    return _IS_IPV6_ENABLED
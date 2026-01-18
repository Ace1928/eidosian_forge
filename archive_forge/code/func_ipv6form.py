from __future__ import absolute_import, division, print_function
from functools import partial
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddr_utils import _need_netaddr
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
@_need_ipaddress
def ipv6form(value, format):
    try:
        if format == 'expand':
            return ip_address(value).exploded
        elif format == 'compress':
            return ip_address(value).compressed
        elif format == 'x509':
            return _handle_x509(value)
    except ValueError:
        msg = 'You must pass a valid IP address; {0} is invalid'.format(value)
        raise AnsibleFilterError(msg)
    if not isinstance(format, str):
        msg = 'You must pass valid format; {0} is not a valid format'.format(format)
        raise AnsibleFilterError(msg)
from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from datetime import datetime
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import (
from ipaddress import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip_interface
from ..module_utils.teem import send_teem
def encode_rd_address(self, record, match, ipv6=False):
    if is_valid_ip_interface(match.group('addr')):
        key = ip_interface(u'{0}/{1}'.format(match.group('addr'), match.group('cidr')))
    else:
        raise F5ModuleError("When specifying an 'address' type, the value to the left of the separator must be an IP.")
    if key and 'value' in record:
        if ipv6 and key.network.prefixlen == 128:
            return self.encode_host(str(key.ip) + '%' + match.group('rd'), record['value'])
        elif not ipv6 and key.network.prefixlen == 32:
            return self.encode_host(str(key.ip) + '%' + match.group('rd'), record['value'])
        return self.encode_network(str(key.network.network_address) + '%' + match.group('rd'), key.network.prefixlen, record['value'])
    elif key:
        if ipv6 and key.network.prefixlen == 128:
            return self.encode_host(str(key.ip) + '%' + match.group('rd'), str(key.ip) + '%' + match.group('rd'))
        elif not ipv6 and key.network.prefixlen == 32:
            return self.encode_host(str(key.ip) + '%' + match.group('rd'), str(key.ip) + '%' + match.group('rd'))
        return self.encode_network(str(key.network.network_address) + '%' + match.group('rd'), key.network.prefixlen, str(key.network.network_address) + '%' + match.group('rd'))
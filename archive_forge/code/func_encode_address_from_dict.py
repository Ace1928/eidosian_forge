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
def encode_address_from_dict(self, record):
    rd_match = re.match(self._ipv4_cidr_ptrn_rd, record['key'])
    if rd_match:
        return self.encode_rd_address(record, rd_match)
    rd_match = re.match(self._ipv6_cidr_ptrn_rd, record['key'])
    if rd_match:
        return self.encode_rd_address(record, rd_match, ipv6=True)
    if is_valid_ip_interface(record['key']):
        key = ip_interface(u'{0}'.format(str(record['key'])))
    else:
        raise F5ModuleError("When specifying an 'address' type, the value to the left of the separator must be an IP.")
    ipv4_match = re.match(self._ipv4_cidr_ptrn, record['key'])
    ipv6_match = re.match(self._ipv6_cidr_ptrn, record['key'])
    if key and 'value' in record:
        if ipv6_match and key.network.prefixlen == 128 or (ipv4_match and key.network.prefixlen == 32):
            return self.encode_host(str(key.ip), record['value'])
        else:
            return self.encode_network(str(key.network.network_address), key.network.prefixlen, record['value'])
    elif key:
        if ipv6_match and key.network.prefixlen == 128 or (ipv4_match and key.network.prefixlen == 32):
            return self.encode_host(str(key.ip), str(key.ip))
        else:
            return self.encode_network(str(key.network.network_address), key.network.prefixlen, str(key.network.network_address))
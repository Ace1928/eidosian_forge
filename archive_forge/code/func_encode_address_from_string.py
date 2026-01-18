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
def encode_address_from_string(self, record):
    if self._network_pattern.match(record):
        return record
    elif self._host_pattern.match(record):
        return record
    elif self._rd_net_prefix_ptrn.match(record) or self._rd_host_ptrn.match(record):
        return record
    elif self._ipv4_cidr_ptrn_rd.match(record) or self._ipv6_cidr_ptrn_rd.match(record):
        parts = [r.strip() for r in record.split(self._separator)]
        if parts[0] == '':
            return
        pattern = re.compile('(?P<addr>[^%]+)%(?P<rd>[0-9]+)/(?P<prefix>[0-9]+)')
        match = pattern.match(parts[0])
        addr = u'{0}/{1}'.format(match.group('addr'), match.group('prefix'))
        if not is_valid_ip_interface(addr):
            raise F5ModuleError("When specifying an 'address' type, the value to the left of the separator must be an IP.")
        key = ip_interface(addr)
        ipv4_match = re.match(self._ipv4_cidr_ptrn, addr)
        ipv6_match = re.match(self._ipv6_cidr_ptrn, addr)
        if len(parts) == 2:
            if ipv4_match and key.network.prefixlen == 32 or (ipv6_match and key.network.prefixlen == 128):
                return self.encode_host(str(key.ip) + '%' + str(match.group('rd')), parts[1])
            else:
                return self.encode_network(str(key.network.network_address) + '%' + str(match.group('rd')), key.network.prefixlen, parts[1])
        elif len(parts) == 1 and parts[0] != '':
            if ipv4_match and key.network.prefixlen == 32 or (ipv6_match and key.network.prefixlen == 128):
                return self.encode_host(str(key.ip) + '%' + str(match.group('rd')), str(key.ip) + '%' + str(match.group('rd')))
            return self.encode_network(str(key.network.network_address) + '%' + str(match.group('rd')), key.network.prefixlen, str(key.network.network_address) + '%' + str(match.group('rd')))
    else:
        parts = [r.strip() for r in record.split(self._separator)]
        if parts[0] == '':
            return
        if len(re.split(' ', parts[0])) == 1:
            if not is_valid_ip_interface(parts[0]):
                raise F5ModuleError("When specifying an 'address' type, the value to the left of the separator must be an IP.")
            key = ip_interface(u'{0}'.format(str(parts[0])))
            ipv4_match = re.match(self._ipv4_cidr_ptrn, str(parts[0]))
            ipv6_match = re.match(self._ipv6_cidr_ptrn, str(parts[0]))
            if len(parts) == 2:
                if ipv4_match and key.network.prefixlen == 32 or (ipv6_match and key.network.prefixlen == 128):
                    return self.encode_host(str(key.ip), parts[1])
                else:
                    return self.encode_network(str(key.network.network_address), key.network.prefixlen, parts[1])
            elif len(parts) == 1 and parts[0] != '':
                if ipv4_match and key.network.prefixlen == 32 or (ipv6_match and key.network.prefixlen == 128):
                    return self.encode_host(str(key.ip), str(key.ip))
                return self.encode_network(str(key.network.network_address), key.network.prefixlen, str(key.network.network_address))
        else:
            return str(parts[0])
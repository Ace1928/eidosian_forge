import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_list_reserved_ipv4(self, vlan=None, datacenter_id=None):
    if vlan is not None:
        vlan_id = self._vlan_to_vlan_id(vlan)
        params = {'vlanId': vlan_id}
        response = self.connection.request_with_orgId_api_2('network/reservedPrivateIpv4Address', params=params).object
    elif datacenter_id is not None:
        params = {'datacenterId': datacenter_id}
        response = self.connection.request_with_orgId_api_2('network/reservedPrivateIpv4Address', params=params).object
    else:
        response = self.connection.request_with_orgId_api_2('network/reservedPrivateIpv4Address').object
    addresses = self._to_ipv4_addresses(response)
    return addresses
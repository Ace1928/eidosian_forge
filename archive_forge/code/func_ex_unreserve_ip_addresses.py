import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_unreserve_ip_addresses(self, vlan, ip):
    vlan_id = self._vlan_to_vlan_id(vlan)
    if re.match('(\\d+\\.){3}', ip):
        private_ip = ET.Element('unreservePrivateIpv4Address', {'xmlns': TYPES_URN})
        resource = 'network/reservePrivateIpv4Address'
    elif re.search(':', ip):
        private_ip = ET.Element('unreserveIpv6Address', {'xmlns': TYPES_URN})
        resource = 'network/unreserveIpv6Address'
    ET.SubElement(private_ip, 'vlanId').text = vlan_id
    ET.SubElement(private_ip, 'ipAddress').text = ip
    result = self.connection.request_with_orgId_api_2(resource, method='POST', data=ET.tostring(private_ip)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']
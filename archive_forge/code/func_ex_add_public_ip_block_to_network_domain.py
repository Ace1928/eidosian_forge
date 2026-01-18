import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_add_public_ip_block_to_network_domain(self, network_domain):
    add_node = ET.Element('addPublicIpBlock', {'xmlns': TYPES_URN})
    ET.SubElement(add_node, 'networkDomainId').text = network_domain.id
    response = self.connection.request_with_orgId_api_2('network/addPublicIpBlock', method='POST', data=ET.tostring(add_node)).object
    block_id = None
    for info in findall(response, 'info', TYPES_URN):
        if info.get('name') == 'ipBlockId':
            block_id = info.get('value')
    return self.ex_get_public_ip_block(block_id)
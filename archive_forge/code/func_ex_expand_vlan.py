import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_expand_vlan(self, vlan):
    """
        Expands the VLAN to the prefix size in private_ipv4_range_size
        The expansion will
        not be permitted if the proposed IP space overlaps with an
        already deployed VLANs IP space.

        :param      vlan: The VLAN to update
        :type       vlan: :class:`NttCisNetworkDomain`

        :return: an instance of `NttCisVlan`
        :rtype: :class:`NttCisVlan`
        """
    edit_node = ET.Element('expandVlan', {'xmlns': TYPES_URN})
    edit_node.set('id', vlan.id)
    ET.SubElement(edit_node, 'privateIpv4PrefixSize').text = vlan.private_ipv4_range_size
    self.connection.request_with_orgId_api_2('network/expandVlan', method='POST', data=ET.tostring(edit_node)).object
    return vlan